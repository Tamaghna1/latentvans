"""Check ground_objects.py's grounding on a few extracted last frames before the full pre-pass: what Qwen
names, and which names Grounding DINO boxes (with scores). History: asked to box everything itself,
Qwen2.5-VL-3B returned generic labels ("object", "food") and missed or invented objects (job 68483)."""
import argparse
import glob
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from ground_objects import Boxer, Lister, parse_names  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--box_threshold", type=float, default=0.3)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()
    lister, boxer = Lister(args.qwen_dir), Boxer(args.gdino, 0.0)
    rows = {json.loads(l)["sample_id"]: json.loads(l) for l in open(os.path.join(args.data_dir, "benchmark.jsonl"))}
    frames = sorted(glob.glob(os.path.join(args.data_dir, "ground", "*_last.jpg")))
    picks = frames[::max(len(frames) // args.n, 1)][:args.n]
    for f in picks:
        sid = os.path.basename(f)[:-len("_last.jpg")]
        r = rows[sid]
        img = Image.open(f).convert("RGB")
        text = lister([img], [r["instruction"]])[0]
        names = parse_names(text)
        objs = boxer(img, names)
        print(f"\n## {sid} {img.size} | now: {r['input_sentence']} | next: {r['target_sentence']}")
        print(f"  named: {names}")
        print("  kept:    " + "; ".join(f"{o['label']} {o['box']} {o['score']}" for o in objs if o["score"] >= args.box_threshold))
        print("  dropped: " + "; ".join(f"{o['label']} {o['score']}" for o in objs if o["score"] < args.box_threshold))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
