"""Visual grounding pre-pass for VANS benchmark / training rows: which objects are where in the last input frame.

For every row: the last decoded frame of the input clip (the last of the 6 frames VANS's VLM samples, here at
the clip's native resolution instead of the VLM's 364x644) -> Qwen2.5-VL-3B-Instruct (the base model, no VANS
LoRA), asked to box every object, tool, ingredient and hand involved in the activity -> labels + boxes on a
0-1000 scale. Also saves a zoomed crop of the active region (hands and the objects they touch; else the
largest object and what touches it) for vans_ground.py's ground_zoom arm.

Output per data dir: ground/{<sample_id>_last.jpg, <sample_id>_zoom.jpg, objects_shard<k>.jsonl}; resumable.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

import torch
from PIL import Image

PROMPT = ('Locate every object, tool, ingredient and hand involved in the activity shown in this image, and '
          'output all the bounding box coordinates in JSON format: a list of {"bbox_2d": [x1, y1, x2, y2], '
          '"label": "<short name with its color or material>"}.')
MAX_OBJECTS = 12
ZOOM_SIDE = 644  # longer side of the saved crop (the VLM's video frames are 364x644)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def last_frame(src, dest):
    """Decode the final second and keep overwriting one file, so it ends as the last decoded frame."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-1", "-i", src, "-update", "1", "-q:v", "2", dest],
                   check=True, capture_output=True)
    return Image.open(dest).convert("RGB")


def parse_objects(text, w_in, h_in):
    """Qwen2.5-VL boxes are in pixels of its resized input (w_in x h_in); returns them on a 0-1000 scale.
    Items are parsed one by one so a list cut off by max_new_tokens still yields its complete items."""
    out, seen = [], set()
    for item in re.findall(r"\{[^{}]*\}", text):
        try:
            d = json.loads(item)
            x1, y1, x2, y2 = [float(v) for v in d["bbox_2d"]]
            label = str(d["label"]).strip()
        except (ValueError, KeyError, TypeError):
            continue
        b = [round(min(max(v / s * 1000, 0), 1000)) for v, s in ((x1, w_in), (y1, h_in), (x2, w_in), (y2, h_in))]
        if b[2] - b[0] < 5 or b[3] - b[1] < 5 or not label or (label.lower(), tuple(b)) in seen:
            continue
        seen.add((label.lower(), tuple(b)))
        out.append({"label": label, "box": b})
    return out[:MAX_OBJECTS]


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1]) / 1e6


def touches(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def fit(lo, hi, min_len=400, pad=100):
    lo, hi = lo - pad, hi + pad
    if hi - lo < min_len:
        c = (lo + hi) / 2
        lo, hi = c - min_len / 2, c + min_len / 2
    if lo < 0:
        lo, hi = 0, hi - lo
    if hi > 1000:
        lo, hi = max(lo - (hi - 1000), 0), 1000
    return round(lo), round(hi)


def zoom_box(objs):
    """Active region on the 0-1000 scale, padded by 10% and at least 40% of the frame per side."""
    boxes = [(o["label"].lower(), o["box"]) for o in objs if area(o["box"]) < 0.5]
    hands = [b for l, b in boxes if "hand" in l]
    if hands:
        sel = hands + [b for l, b in boxes if "hand" not in l and any(touches(b, h) for h in hands)]
    elif boxes:
        big = max(boxes, key=lambda x: area(x[1]))[1]
        sel = [big] + [b for _, b in boxes if b is not big and touches(b, big)]
    else:
        return [250, 250, 750, 750]
    x1, x2 = fit(min(b[0] for b in sel), max(b[2] for b in sel))
    y1, y2 = fit(min(b[1] for b in sel), max(b[3] for b in sel))
    return [x1, y1, x2, y2]


def save_zoom(img, box, dest):
    W, H = img.size
    crop = img.crop((box[0] * W / 1000, box[1] * H / 1000, box[2] * W / 1000, box[3] * H / 1000))
    s = ZOOM_SIDE / max(crop.size)
    crop.resize((round(crop.width * s), round(crop.height * s)), Image.BICUBIC).save(dest, quality=92)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="first N rows of each data dir (smoke test)")
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    proc = AutoProcessor.from_pretrained(args.qwen_dir, max_pixels=1280 * 28 * 28)
    proc.tokenizer.padding_side = "left"

    for d in args.data_dirs:
        gdir = os.path.join(d, "ground")
        os.makedirs(gdir, exist_ok=True)
        rows = [json.loads(l) for l in open(os.path.join(d, "benchmark.jsonl"))]
        rows = (rows[:args.limit] if args.limit else rows)[args.shard::args.num_shards]
        path = os.path.join(gdir, f"objects_shard{args.shard}.jsonl")
        done = set()
        for f in glob.glob(os.path.join(gdir, "objects_shard*.jsonl")):
            done |= {json.loads(l)["sample_id"] for l in open(f)}
        todo = [r for r in rows if r["sample_id"] not in done]
        log(f"{d}: {len(rows)} rows in shard {args.shard}, {len(todo)} to do")
        n_obj, t0 = 0, time.time()
        for i in range(0, len(todo), args.batch_size):
            batch, imgs = [], []
            for r in todo[i:i + args.batch_size]:
                try:
                    imgs.append(last_frame(os.path.join(d, r["input_video"]),
                                           os.path.join(gdir, f"{r['sample_id']}_last.jpg")))
                    batch.append(r)
                except Exception as e:
                    log(f"WARNING: {r['sample_id']}: last frame failed ({str(e)[-200:]})")
            if not batch:
                continue
            msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]] * len(batch)
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inp = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
            outs = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            with open(path, "a") as f:
                for r, img, text, g in zip(batch, imgs, outs, inp["image_grid_thw"].tolist()):
                    objs = parse_objects(text, g[2] * 14, g[1] * 14)
                    zb = zoom_box(objs)
                    save_zoom(img, zb, os.path.join(gdir, f"{r['sample_id']}_zoom.jpg"))
                    n_obj += len(objs)
                    f.write(json.dumps({"sample_id": r["sample_id"], "frame": f"ground/{r['sample_id']}_last.jpg",
                                        "zoom": f"ground/{r['sample_id']}_zoom.jpg", "zoom_box": zb,
                                        "size": list(img.size), "objects": objs, "raw": text}) + "\n")
            k = min(i + args.batch_size, len(todo))
            log(f"{os.path.basename(d)} shard {args.shard}: {k}/{len(todo)}, {n_obj / k:.1f} objects/frame, "
                f"{(time.time() - t0) / k:.2f} s/row | e.g. {[o['label'] for o in objs][:6]}")
    log("GROUNDING DONE")


if __name__ == "__main__":
    main()
