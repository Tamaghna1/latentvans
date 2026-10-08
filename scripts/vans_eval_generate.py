"""Generate VANS-benchmark answers (caption + video) with the released VANS model.

Arms:
  vans         VANS VLM writes the caption from (input video, instruction), VANS VDM renders it --
               the paper's full system, used to check our benchmark/metrics against its Table 1.
  vans_oracle  VANS VDM rendering the ground-truth caption (no VLM): the gap between this and
               `vans` is the part of the error due to reasoning rather than the video model.

Inference settings follow code/VANS/inference.py: 352x640, 33 frames, 50 steps, CFG 5, shift 5,
seed 0, their negative prompt, 6 reference latents from the input video. Two departures, both
for disk/env reasons: the umT5 text encoder is HF UMT5EncoderModel loaded from our diffusers
checkpoint (the same weights DiffSynth loads from the original .pth, same zero padding past
the prompt length), and the DiT is built directly from VANS_vlm.safetensors (a full DiT
state dict) rather than from base Wan weights that it would overwrite anyway.

Output: <out_dir>/<arm>/{videos/<sample_id>.mp4, captions.jsonl}; resumable, shardable.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

NEGATIVE = ("vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, "
            "stationary, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed "
            "limbs, fused fingers, stationary frame, cluttered background, three legs, crowded background "
            "people, walking backwards")
H, W, F = 352, 640, 33


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_frames(path, n=F, h=H, w=W):
    """n evenly spaced frames of the clip, resized to cover h x w and center-cropped."""
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out = []
    for i in idx:
        im = Image.fromarray(frames[i])
        s = max(h / im.height, w / im.width)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        l, t = (im.width - w) // 2, (im.height - h) // 2
        out.append(im.crop((l, t, l + w, t + h)))
    return out


class HFUMT5:
    """Stands in for DiffSynth's WanTextEncoder: same call signature (ids, mask) -> (B, L, 4096).
    Lives on the CPU and visits the GPU only while encoding (11 GB), so VLM + DiT + VAE + umT5 fit
    on a 48 GB card."""

    def __init__(self, wan_diffusers_dir, device):
        from transformers import UMT5EncoderModel
        self.model = UMT5EncoderModel.from_pretrained(wan_diffusers_dir, subfolder="text_encoder",
                                                      torch_dtype=torch.bfloat16).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, ids, mask):
        self.model.to(self.device)
        try:
            return self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
        finally:
            self.model.to("cpu")
            torch.cuda.empty_cache()

    def to(self, *a, **k):
        return self

    def parameters(self):
        return self.model.parameters()


def build_pipe(args, use_mllm):
    from vans_rl_common import vans_imports
    vans_imports(args.vans_code)
    from safetensors.torch import load_file
    from vans.models.model_manager import ModelManager
    from vans.models.wan_video_dit import WanModel
    from vans.pipelines.wan_video_new import WanVideoPipeline
    device = "cuda"

    pipe = WanVideoPipeline(device=device, torch_dtype=torch.bfloat16)
    # modules WanVideoPipeline.from_pretrained would set (to None, for a T2V model); its units read them
    for name in ("dit2", "image_encoder", "motion_controller", "vace", "vace2", "audio_encoder", "audio_processor"):
        setattr(pipe, name, None)
    mm = ModelManager()
    mm.load_model(os.path.join(args.wan_orig_dir, "Wan2.1_VAE.pth"), device=device, torch_dtype=torch.bfloat16)
    pipe.vae = mm.fetch_model("wan_video_vae")
    pipe.height_division_factor = pipe.vae.upsampling_factor * 2
    pipe.width_division_factor = pipe.vae.upsampling_factor * 2

    sd = load_file(os.path.join(args.vans_dir, "VANS_vlm.safetensors"))
    dit_sd = {k[len("dit."):] if k.startswith("dit.") else k: v for k, v in sd.items() if ".mllm." not in k}
    dit = WanModel(has_image_input=False, patch_size=[1, 2, 2], in_dim=16, dim=1536, ffn_dim=8960, freq_dim=256,
                   text_dim=4096, out_dim=16, num_heads=12, num_layers=30, eps=1e-6)
    missing, unexpected = dit.load_state_dict(dit_sd, strict=False)
    log(f"VANS DiT: {len(dit_sd)} tensors, missing {len(missing)} {missing[:5]}, unexpected {len(unexpected)} {unexpected[:5]}")
    if len(missing) > 0:
        raise RuntimeError("VANS DiT checkpoint does not cover the 1.3B WanModel; check key format")
    pipe.dit = dit.to(device=device, dtype=torch.bfloat16).eval()

    pipe.text_encoder = HFUMT5(args.wan_diffusers_dir, device)
    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(os.path.join(args.wan_orig_dir, "google/umt5-xxl"))

    if use_mllm:
        from vans.models_mllm.agent_mllm import MLLMAgent
        pipe.mllm = MLLMAgent(mllm_pretrained_path=args.qwen_dir).to(device)
        msd = load_file(os.path.join(args.vans_dir, "VANS_mllm.safetensors"))
        own = pipe.state_dict()
        msd = {k.replace("pipe.", ""): v for k, v in msd.items()}
        hit = {k: v for k, v in msd.items() if k in own}
        log(f"VANS VLM LoRA: {len(hit)}/{len(msd)} tensors matched")
        if len(hit) != len(msd):
            raise RuntimeError(f"unmatched VLM keys, e.g. {[k for k in msd if k not in own][:5]}")
        pipe.load_state_dict(hit, strict=False)
        pipe.mllm.eval()
    return pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["vans", "vans_oracle"], required=True)
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vans_code", required=True)
    p.add_argument("--vans_dir", required=True)
    p.add_argument("--wan_orig_dir", required=True)
    p.add_argument("--wan_diffusers_dir", required=True)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from vans_rl_common import vans_imports
    vans_imports(args.vans_code)
    from vans import save_video
    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    rows = rows[:args.limit] if args.limit else rows
    rows = rows[args.shard::args.num_shards]
    out = os.path.join(args.out_dir, args.arm)
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    cap_path = os.path.join(out, f"captions_shard{args.shard}.jsonl")
    done = {json.loads(l)["sample_id"] for l in open(cap_path)} if os.path.exists(cap_path) else set()

    pipe = build_pipe(args, use_mllm=args.arm == "vans")
    for i, r in enumerate(rows):
        if r["sample_id"] in done:
            continue
        t0 = time.time()
        in_path = os.path.join(args.bench_dir, r["input_video"])
        ref = load_frames(in_path)
        if args.arm == "vans":
            if i == 0:
                probe, _, _ = pipe.mllm.tokenize(pipe.mllm.tokenizer, is_train=False, add_query=False,
                                                 instructions=r["instruction"], caption="", video=in_path,
                                                 device="cuda", torch_dtype=torch.bfloat16)
                log(f"VLM input: {probe['input_ids'].shape[1]} tokens, video grid {probe['video_grid_thw'].tolist()}")
            torch.manual_seed(0)
            text, video = pipe(prompt="", instructions=r["instruction"], input_video_path=in_path,
                               negative_prompt=NEGATIVE, seed=0, tiled=True, height=H, width=W,
                               num_frames=F, ref_VAE_video=ref)
            from vans.pipelines.wan_video_new import extract_caption
            caption = extract_caption(text)
        else:
            text = None
            caption = r["gt_caption"]
            video = pipe(prompt=caption, negative_prompt=NEGATIVE, seed=0, tiled=True, height=H, width=W,
                         num_frames=F, ref_VAE_video=ref)
        save_video(video, os.path.join(out, "videos", f"{r['sample_id']}.mp4"), fps=11, quality=5)
        with open(cap_path, "a") as f:
            f.write(json.dumps({"sample_id": r["sample_id"], "caption": caption, "raw": text}) + "\n")
        log(f"[{args.arm} shard {args.shard}] {i + 1}/{len(rows)} {r['sample_id']} ({time.time() - t0:.0f}s)")
    log("GENERATION DONE")


if __name__ == "__main__":
    main()
