"""Scene retention of answer videos (plan.md P27 (b)): how much does the answer video look like the last input frame?
Replaces the judge's p_scene question, which the 3B judge answers ~0.5 for every arm.

Per sample, the last input frame and 33 evenly spaced answer frames (both at 352x640, resize-to-cover + crop, as
in vans_eval_metrics.py) are embedded and compared by cosine, in the spirit of VBench's consistency dimensions
(DINO features for subject consistency, CLIP for background consistency), but against the input instead of the
video's own first frame:
  dino      DINOv2 ViT-S/14 CLS cosine, mean over the 33 frames (layout and objects)
  dino_end  the same over the last 8 frames (does the video drift out of the scene?)
  clip      CLIP ViT-B/32 image cosine, mean over the 33 frames (semantic scene)
  clip_end  the same over the last 8 frames
The real next clip (arm 'target_real') gives the natural level: the true next step can move the camera or the
place, so the goal is to be near the target, not at the frozen frame's ~1.0.

  python scene_sim.py --bench_dir data/vans_eval --results_dir results/vans_eval --arms target_real vans copy_last
Output: <results_dir>/<arm>/scene.json (overall, per source, per sample) and one row per arm appended to
<results_dir>/scene_summary.jsonl. Arms with scene.json are skipped unless --force. With --limit (smoke) the output
is scene_smoke.json and nothing is appended or logged. Frames go through each model's own processor (resize, center
crop 224), as CLIP-V does.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from vans_eval_metrics import Clip, read_video  # noqa: E402

DINO = "facebook/dinov2-small"
END = 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Dino:
    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        self.m = AutoModel.from_pretrained(DINO).to(device).eval()
        self.p = AutoImageProcessor.from_pretrained(DINO)
        self.device = device

    @torch.no_grad()
    def images(self, frames):
        x = self.p(images=[Image.fromarray(f) for f in frames], return_tensors="pt")["pixel_values"]
        return F.normalize(self.m(pixel_values=x.to(self.device)).pooler_output, dim=-1)


def summarize(per):
    out = {}
    for src in (None, "coin", "youcook"):
        rows = [p for p in per if src is None or p["source"] == src]
        if rows:
            k = src or "overall"
            out[f"{k}/n"] = len(rows)
            for q in ("dino", "dino_end", "clip", "clip_end"):
                out[f"{k}/{q}"] = float(np.mean([p[q] for p in rows]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="arm dirs under --results_dir, or 'target_real'")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    rows = rows[:args.limit] if args.limit else rows
    clip, dino = Clip(device), Dino(device)
    anchor = {}  # sample_id -> (dino, clip) embedding of the last input frame

    for arm in args.arms:
        arm_dir = os.path.join(args.results_dir, arm)
        out_path = os.path.join(arm_dir, "scene_smoke.json" if args.limit else "scene.json")
        if os.path.exists(out_path) and not (args.force or args.limit):
            log(f"{arm}: scene.json exists, skipped")
            continue

        def video_path(r):
            return (os.path.join(args.bench_dir, r["target_video"]) if arm == "target_real"
                    else os.path.join(arm_dir, "videos", f"{r['sample_id']}.mp4"))

        mine = [r for r in rows if os.path.exists(video_path(r))]
        if not mine:
            log(f"WARNING: {arm}: no videos, skipped")
            continue
        log(f"{arm}: {len(mine)} videos")
        per = []
        for r in mine:
            sid = r["sample_id"]
            if sid not in anchor:
                last = read_video(os.path.join(args.bench_dir, r["input_video"]))[-1:]
                anchor[sid] = (dino.images(last), clip.images(last))
            frames = read_video(video_path(r))
            sd = (dino.images(frames) @ anchor[sid][0].T).squeeze(1)
            sc = (clip.images(frames) @ anchor[sid][1].T).squeeze(1)
            per.append({"sample_id": sid, "source": r["source"], "dino": float(sd.mean()),
                        "dino_end": float(sd[-END:].mean()), "clip": float(sc.mean()), "clip_end": float(sc[-END:].mean())})
        summ = summarize(per)
        log(f"{arm} " + " ".join(f"{k}={v:.3f}" for k, v in summ.items() if k.startswith("overall/")))
        os.makedirs(arm_dir, exist_ok=True)
        json.dump({"arm": arm, "dino_model": DINO, **summ, "per_sample": per}, open(out_path, "w"), indent=1)
        if args.limit:
            continue
        with open(os.path.join(args.results_dir, "scene_summary.jsonl"), "a") as f:
            f.write(json.dumps({"arm": arm, "dino_model": DINO, **summ, "time": time.time()}) + "\n")
        from wandb_util import log_summary
        log_summary(args.wandb_project, f"scene-{arm}", {**vars(args), "arm": arm}, summ, job_type="scene_sim")
    log("SCENE SIM DONE")


if __name__ == "__main__":
    main()
