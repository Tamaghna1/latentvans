"""Precompute everything the VDM pilot trains on, so training only needs the Wan DiT.

--what video   For each pilot example, build a 17-frame clip at --height x --width:
               frame 0 is the context frame (TwiFF question index), frames 1..16 are
               spaced evenly from just after it up to the last reasoning frame -- i.e.
               the future the question asks about, time-lapsed to 16 frames. Encode it
               with the Wan VAE -> (16, 5, H/8, W/8) latent (frame 0 alone becomes latent
               frame 0, since the Wan VAE is causal), normalized with the VAE's
               latents_mean/std exactly as WanPipeline expects.
--what text    Encode captions from vdm_pilot_vlm_outputs.py with Wan's umT5 text
               encoder; keep only the real (unpadded) tokens, plus the empty-prompt
               embedding for the "null" arm.

Keys match vdm_pilot_vlm_outputs.py ("<video>|<row index>"). Outputs go to
<output_dir>/{split}_video.pt and <output_dir>/{split}_text_<tag>.pt. Resumable.
"""
import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twiff_frames import count_decoded_frames, read_frames_resized, twiff_index_to_frame  # noqa: E402

N_FUTURE = 16


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", choices=["video", "text"], required=True)
    p.add_argument("--wan_dir", required=True)
    p.add_argument("--train_metadata_dir", required=True)
    p.add_argument("--train_rows_json", required=True)
    p.add_argument("--train_video_root", required=True)
    p.add_argument("--val_metadata_dir", required=True)
    p.add_argument("--val_video_root", required=True)
    p.add_argument("--val_frames_root", required=True)
    p.add_argument("--vlm_dir", help="--what text: vdm_pilot_vlm_outputs.py output dir holding the captions")
    p.add_argument("--text_tag", default="caption", help="--what text: name for the output file")
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--splits", default="val,train")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--decode_workers", type=int, default=8, help="parallel ffmpeg decodes feeding the GPU")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def split_rows(args, split):
    from datasets import load_from_disk

    if split == "train":
        ds = load_from_disk(args.train_metadata_dir)
        rows, root = json.load(open(args.train_rows_json))["rows"], args.train_video_root
    else:
        ds = load_from_disk(args.val_metadata_dir)
        have = set(os.listdir(args.val_frames_root))
        rows, root = [i for i, v in enumerate(ds["video"]) if v[:-4] in have], args.val_video_root
    return ds, (rows[:args.limit] if args.limit else rows), root


def frame_plan(q_idx, r_idx, n):
    """0-based frame numbers: [context] + N_FUTURE evenly spaced frames up to the last reasoning frame."""
    ctx = twiff_index_to_frame(max(q_idx), n)
    end = twiff_index_to_frame(max(r_idx), n) if r_idx else n - 1
    if end <= ctx:
        end = n - 1
    span = max(end - ctx, 1)
    return [ctx] + [min(ctx + round(k * span / N_FUTURE), n - 1) for k in range(1, N_FUTURE + 1)]


@torch.no_grad()
def encode_videos(args):
    from diffusers import AutoencoderKLWan

    device = torch.device("cuda")
    vae = AutoencoderKLWan.from_pretrained(args.wan_dir, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)
    for split in args.splits.split(","):
        out_path = os.path.join(args.output_dir, f"{split}_video.pt")
        st = torch.load(out_path) if os.path.exists(out_path) else {"keys": [], "latents": [], "plans": []}
        if isinstance(st["latents"], torch.Tensor):
            st["latents"] = list(st["latents"].unbind(0))
        done = set(st["keys"])
        ds, rows, root = split_rows(args, split)
        log(f"{split}: {len(rows)} rows, {len(done)} done")
        n_fail = 0
        todo = [i for i in rows if f"{ds[i]['video']}|{i}" not in done]

        def decode(i):
            """CPU side (runs in a thread pool): ffmpeg count + decode for one row."""
            row = ds[i]
            path = os.path.join(root, row["video"])
            n = count_decoded_frames(path)
            plan = frame_plan(row["question_images_index"], row["reasoning_images_index"], n)
            return row, plan, read_frames_resized(path, plan, args.width, args.height)

        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=args.decode_workers)
        from collections import deque
        window = deque()  # at most 4 x decode_workers decoded-or-pending clips in memory
        next_submit = 0
        for n_i, i in enumerate(todo):
            while next_submit < len(todo) and len(window) < 4 * args.decode_workers:
                window.append(pool.submit(decode, todo[next_submit]))
                next_submit += 1
            fut = window.popleft()
            key = f"{ds[i]['video']}|{i}"
            try:
                row, plan, frames = fut.result()
            except Exception as e:
                n_fail += 1
                log(f"  skip {ds[i]['video']}: {e}")
                continue
            video = torch.stack([torch.from_numpy(frames[f].copy()) for f in plan])  # (17, H, W, 3)
            video = video.permute(3, 0, 1, 2).unsqueeze(0).float().div(127.5).sub(1.0).to(device)
            z = vae.encode(video).latent_dist.mode()
            st["latents"].append(((z - mean) / std)[0].to(torch.bfloat16).cpu())
            st["keys"].append(key)
            st["plans"].append(plan)
            if (n_i + 1) % 100 == 0:
                log(f"  {split}: {len(done) + n_i + 1}/{len(rows)} (failed {n_fail})")
            if (n_i + 1) % args.save_every == 0:
                torch.save({**st, "latents": torch.stack(st["latents"])}, out_path)
        pool.shutdown()
        torch.save({**st, "latents": torch.stack(st["latents"]), "height": args.height, "width": args.width},
                   out_path)
        log(f"{split}: saved {len(st['keys'])} latents ({n_fail} failed) to {out_path}")


@torch.no_grad()
def encode_texts(args):
    from transformers import AutoTokenizer, UMT5EncoderModel

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.wan_dir, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(args.wan_dir, subfolder="text_encoder",
                                           torch_dtype=torch.bfloat16).to(device).eval()

    def embed(texts):
        t = tok([re.sub(r"\s+", " ", x).strip() for x in texts], padding="max_length", max_length=512,
                truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        h = enc(t.input_ids.to(device), t.attention_mask.to(device)).last_hidden_state
        lens = t.attention_mask.sum(1).tolist()
        return [h[j, :lens[j]].to(torch.bfloat16).cpu() for j in range(len(texts))]

    null = embed([""])[0]
    for split in args.splits.split(","):
        vlm = torch.load(os.path.join(args.vlm_dir, f"{split}_vlm.pt"))
        caps = vlm["captions"]
        embs = []
        for s in range(0, len(caps), 32):
            embs += embed(caps[s:s + 32])
            if (s // 32) % 20 == 0:
                log(f"  {split}: {s + len(caps[s:s + 32])}/{len(caps)}")
        out_path = os.path.join(args.output_dir, f"{split}_text_{args.text_tag}.pt")
        torch.save({"keys": vlm["keys"], "embeds": embs, "null": null, "captions": caps}, out_path)
        log(f"{split}: saved {len(embs)} caption embeddings to {out_path} "
            f"(mean length {sum(e.shape[0] for e in embs) / max(len(embs), 1):.1f} tokens)")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    encode_videos(args) if args.what == "video" else encode_texts(args)


if __name__ == "__main__":
    main()
