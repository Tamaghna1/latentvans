"""Generate VANS-benchmark videos with a VDM pilot arm (train_vdm_pilot.py checkpoint).

The pilot arms were trained on Future-L1-50K at 288x512: latent frame 0 is the context frame
(kept clean, per-token timestep 0) and the 16 frames after it are the future. On the VANS benchmark
the context frame is the last frame of the input clip, and the caption comes from --caption_source:
  null    empty-prompt embedding (no reasoning)
  oracle  the ground-truth caption (benchmark gt_caption)
  <path>  a captions*.jsonl file (e.g. another arm's VLM captions), so two arms render the same text

Sampling: Euler on the rectified flow, --steps steps with Wan's shifted schedule (--shift, the
training shift by default), classifier-free guidance against the null embedding (the arms were trained
with 10% conditioning dropout to null). Output: 16 future frames at 288x512, 8 fps. This is a
zero-shot transfer (different data, resolution, context length from VANS); metrics resample frames.

Output: <out_dir>/<name>/{videos/<sample_id>.mp4, captions_shard<k>.jsonl}; resumable, shardable.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MAX_TEXT_LEN = 512


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def last_frame(path, w, h):
    """Last displayed frame, scaled exactly like the pilot's training frames (ffmpeg bicubic)."""
    from twiff_frames import count_decoded_frames, read_frames_resized
    n = count_decoded_frames(path)
    return read_frames_resized(path, [n - 1], w, h)[n - 1]


def write_mp4(frames, path, fps):
    """(F, H, W, 3) uint8 -> h264 mp4 via the env's ffmpeg (envs/latentvans has no OpenCV)."""
    import subprocess
    from twiff_frames import ffmpeg_exe
    f, h, w, _ = frames.shape
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--ckpt", required=True, help="checkpoints/vdm_pilot/<arm>/step_N")
    p.add_argument("--caption_source", required=True)
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wan_dir", required=True)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=17)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    device = torch.device("cuda")

    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from peft import LoraConfig, set_peft_model_state_dict
    from transformers import AutoTokenizer, UMT5EncoderModel

    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    rows = rows[:args.limit] if args.limit else rows
    rows = rows[args.shard::args.num_shards]
    if args.caption_source == "null":
        captions = {r["sample_id"]: "" for r in rows}
    elif args.caption_source == "oracle":
        captions = {r["sample_id"]: r["gt_caption"] for r in rows}
    else:
        captions = {}
        for f in sorted(glob.glob(args.caption_source)):
            for l in open(f):
                c = json.loads(l)
                captions[c["sample_id"]] = c["caption"]
        missing = [r["sample_id"] for r in rows if r["sample_id"] not in captions]
        if missing:
            log(f"WARNING: {len(missing)} samples have no caption in {args.caption_source}; skipping them")
        rows = [r for r in rows if r["sample_id"] in captions]

    out = os.path.join(args.out_dir, args.name)
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    cap_path = os.path.join(out, f"captions_shard{args.shard}.jsonl")
    done = {json.loads(l)["sample_id"] for l in open(cap_path)} if os.path.exists(cap_path) else set()
    json.dump(vars(args), open(os.path.join(out, "generate_config.json"), "w"), indent=2)

    tok = AutoTokenizer.from_pretrained(args.wan_dir, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(args.wan_dir, subfolder="text_encoder",
                                           torch_dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def embed(text):
        t = tok([re.sub(r"\s+", " ", text).strip()], padding="max_length", max_length=MAX_TEXT_LEN, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        h = enc(t.input_ids.to(device), t.attention_mask.to(device)).last_hidden_state[0]
        n = int(t.attention_mask.sum())
        return F.pad(h[:n].float(), (0, 0, 0, MAX_TEXT_LEN - n))

    null_emb = embed("")

    model = WanTransformer3DModel.from_pretrained(args.wan_dir, subfolder="transformer", torch_dtype=torch.bfloat16)
    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    model.add_adapter(LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                 target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    res = set_peft_model_state_dict(model, torch.load(os.path.join(args.ckpt, "wan_lora.pt"), map_location="cpu"))
    if getattr(res, "unexpected_keys", None):
        raise RuntimeError(f"LoRA keys did not load: {res.unexpected_keys[:5]}")
    model.to(device).eval()
    patch = tuple(model.config.patch_size)

    vae = AutoencoderKLWan.from_pretrained(args.wan_dir, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)

    lat_f = (args.num_frames - 1) // 4 + 1
    lat_h, lat_w = args.height // 8, args.width // 8
    tokens_per_frame = (lat_h // patch[1]) * (lat_w // patch[2])
    u = torch.linspace(1.0, 0.0, args.steps + 1)
    sigmas = args.shift * u / (1 + (args.shift - 1) * u)

    for i, r in enumerate(rows):
        sid = r["sample_id"]
        if sid in done:
            continue
        t0 = time.time()
        with torch.no_grad():
            ctx = last_frame(os.path.join(args.bench_dir, r["input_video"]), args.width, args.height)
            # (1, 3, 1, H, W); Wan's VAE is causal, so this equals latent frame 0 of the full clip
            x = torch.from_numpy(ctx).permute(2, 0, 1)[None, :, None].float().div(127.5).sub(1.0).to(device)
            z0 = (vae.encode(x).latent_dist.mode() - mean) / std  # (1, 16, 1, h, w)
            g = torch.Generator(device="cpu").manual_seed(args.seed * 1_000_003 + i)
            lat = torch.randn((1, 16, lat_f, lat_h, lat_w), generator=g).to(device)
            cond = embed(captions[sid])[None]
            for k in range(args.steps):
                s, s_next = sigmas[k].item(), sigmas[k + 1].item()
                lat[:, :, 0:1] = z0
                t = torch.full((1, lat_f * tokens_per_frame), s * 1000, device=device)
                t[:, :tokens_per_frame] = 0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    both = model(hidden_states=torch.cat([lat, lat]).to(torch.bfloat16), timestep=torch.cat([t, t]),
                                 encoder_hidden_states=torch.cat([cond, null_emb[None]]).to(torch.bfloat16),
                                 return_dict=False)[0].float()
                v = both[1:] + args.cfg * (both[:1] - both[1:])
                lat = lat + (s_next - s) * v
            lat[:, :, 0:1] = z0
            video = vae.decode(lat * std + mean).sample[0]  # (3, F, H, W) in [-1, 1]
            frames = ((video.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()[1:]
        write_mp4(frames, os.path.join(out, "videos", f"{sid}.mp4"), fps=8)
        with open(cap_path, "a") as f:
            f.write(json.dumps({"sample_id": sid, "caption": captions[sid]}) + "\n")
        log(f"[{args.name} shard {args.shard}] {i + 1}/{len(rows)} {sid} ({time.time() - t0:.0f}s)")
    log("GENERATION DONE")


if __name__ == "__main__":
    main()
