"""Controls for the scene-copy baselines (copy_baselines.py, plan.md P16): does the frozen frame beat the
models on CLIP-V only because real frames are sharper than generated ones? Each arm degrades the copy the way
the VANS video model's output is degraded, then vans_eval_metrics.py scores it like any generated arm:

  copy_last_q5    the last input frame held still for 33 frames, saved like generated videos (save_video at
                  11 fps, quality 5) instead of near-lossless x264: compression only
  copy_last_vae   the same frozen clip encoded and decoded by the Wan2.1 VAE (tiled, bf16, as in generation),
                  then saved the same way: the VAE's softening + compression, i.e. what a video model that
                  copied the frame perfectly would output
  copy_input_vae  the input clip (33 evenly spaced frames) through the same round trip

Frames are resized to cover 352x640 and center-cropped exactly as the generator prepares its reference frames.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from vans_eval_generate import F, H, W, load_frames  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cover_crop(im, h=H, w=W):
    s = max(h / im.height, w / im.width)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    l, t = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((l, t, l + w, t + h))


@torch.no_grad()
def vae_round_trip(vae, frames):
    """PIL frames -> (C, T, H, W) in [-1, 1] -> Wan VAE encode + decode -> PIL frames."""
    x = torch.from_numpy(np.stack([np.asarray(f) for f in frames])).float() / 127.5 - 1
    x = x.permute(3, 0, 1, 2).to(torch.bfloat16)
    z = vae.encode([x], "cuda", tiled=True)
    y = vae.decode(z, "cuda", tiled=True)[0]
    y = ((y.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    return [Image.fromarray(f) for f in y]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vans_code", required=True)
    p.add_argument("--wan_orig_dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from vans_rl_common import vans_imports
    vans_imports(args.vans_code)
    from vans import save_video
    from vans.models.model_manager import ModelManager
    mm = ModelManager()
    mm.load_model(os.path.join(args.wan_orig_dir, "Wan2.1_VAE.pth"), device="cuda", torch_dtype=torch.bfloat16)
    vae = mm.fetch_model("wan_video_vae")

    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    rows = rows[:args.limit] if args.limit else rows
    arms = ("copy_last_q5", "copy_last_vae", "copy_input_vae")
    for a in arms:
        os.makedirs(os.path.join(args.out_dir, a, "videos"), exist_ok=True)
    t0 = time.time()
    for k, r in enumerate(rows):
        sid = r["sample_id"]
        dest = {a: os.path.join(args.out_dir, a, "videos", f"{sid}.mp4") for a in arms}
        if all(os.path.exists(d) for d in dest.values()):
            continue
        frame = os.path.join(args.bench_dir, "ground", f"{sid}_last.jpg")  # from ground_objects.py
        if not os.path.exists(frame):  # copy_baselines.py extracts it here when missing
            frame = os.path.join(args.out_dir, "copy_last", "videos", f"{sid}_last.jpg")
        still = [cover_crop(Image.open(frame).convert("RGB"))] * F
        save_video(still, dest["copy_last_q5"], fps=11, quality=5)
        save_video(vae_round_trip(vae, still), dest["copy_last_vae"], fps=11, quality=5)
        clip = load_frames(os.path.join(args.bench_dir, r["input_video"]))
        save_video(vae_round_trip(vae, clip), dest["copy_input_vae"], fps=11, quality=5)
        if (k + 1) % 25 == 0 or k == 0:
            log(f"{k + 1}/{len(rows)} ({(time.time() - t0) / (k + 1):.1f} s/row)")
    log("VAE COPY BASELINES DONE")


if __name__ == "__main__":
    main()
