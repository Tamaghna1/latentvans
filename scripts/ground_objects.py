"""Visual grounding pre-pass for VANS benchmark / training rows: which objects are where in the last input frame.

For every row, on the last decoded frame of the input clip (the last of the 6 frames VANS's VLM samples):
  1. name: Qwen2.5-VL-3B-Instruct (base, no VANS LoRA), given the user's instruction for task context, lists
     the hands and the specific objects, tools, food and ingredients in the scene ("wooden spoon",
     "stainless steel pot", ...). Asked to box everything itself, the 3B model mostly returned generic labels
     ("food", "object") and missed or invented objects (ground_prompt_probe.py, job 68483), but it names well.
  2. box: Grounding DINO boxes each name in one forward pass over "name1. name2. ... hand."; a name keeps
     its best-scoring box if that score clears --box_threshold, else it is dropped (this also filters names
     the VLM invented). Up to two hand boxes.
Boxes are on a 0-1000 scale. Also saves a zoomed crop of the active region (hands and the objects they touch;
else the largest object and what touches it) for vans_ground.py's ground_zoom arm.

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

LIST_PROMPT = ('The person in this image says: "{instruction}"\n'
               'List the specific objects, tools, food and ingredients visible in this image that are involved in '
               'the task. Give each a short specific name with its color or material, like "wooden spoon", '
               '"chopped red onion" or "stainless steel pot". Answer with a comma-separated list only.')
MAX_NAMES = 10
MAX_OBJECTS = 12
ZOOM_SIDE = 644  # longer side of the saved crop (the VLM's video frames are 364x644)
GENERIC = {"food", "object", "objects", "ingredient", "ingredients", "tool", "tools", "item", "items", "thing",
           "hand", "hands", "person", "kitchen", "table", "background"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def last_frame(src, dest):
    """Decode the final second and keep overwriting one file, so it ends as the last decoded frame."""
    subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-v", "error", "-y", "-sseof", "-1", "-i", src,
                    "-update", "1", "-q:v", "2", dest], check=True, capture_output=True)
    return Image.open(dest).convert("RGB")


def parse_names(text):
    """Comma/newline-separated names -> up to MAX_NAMES distinct, non-generic names."""
    out = []
    for n in re.split(r"[,\n;]", text):
        n = re.sub(r"^[\s\-\*\d\.\)]+|[\s\.\"']+$", "", n).strip().strip('"').lower()
        if n and len(n) <= 40 and n not in GENERIC and n not in out:
            out.append(n)
    return out[:MAX_NAMES]


class Lister:
    """Qwen2.5-VL-3B-Instruct naming the objects in a batch of frames."""

    def __init__(self, qwen_dir):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.proc = AutoProcessor.from_pretrained(qwen_dir, max_pixels=1280 * 28 * 28)
        self.proc.tokenizer.padding_side = "left"

    @torch.no_grad()
    def __call__(self, imgs, instructions):
        texts = [self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": LIST_PROMPT.format(instruction=ins)}]}],
            tokenize=False, add_generation_prompt=True) for ins in instructions]
        inp = self.proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
        gen = self.model.generate(**inp, max_new_tokens=96, do_sample=False)
        return self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)


class Boxer:
    """Grounding DINO: best box per name in one pass (per-name score = max sigmoid over the name's tokens)."""

    def __init__(self, model_id, box_threshold):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda").eval()
        self.th = box_threshold

    @torch.no_grad()
    def __call__(self, img, names):
        queries = names + ["hand"]
        text = ". ".join(queries) + "."
        inp = self.proc(images=img, text=text, return_tensors="pt").to("cuda")
        out = self.model(**inp)
        prob = out.logits[0].sigmoid()  # (queries, text tokens)
        boxes = out.pred_boxes[0]  # (queries, 4) normalized cx, cy, w, h
        offsets = self.proc.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        objs, start = [], 0
        for q in queries:
            span = (start, start + len(q))
            start += len(q) + 2
            toks = [i for i, (a, b) in enumerate(offsets) if b > a and a >= span[0] and b <= span[1]]
            if not toks:
                continue
            score = prob[:, toks].max(-1).values
            picks = score.argsort(descending=True)[:2 if q == "hand" else 1]
            for k in picks.tolist():
                s = float(score[k])
                if s < self.th:
                    continue
                cx, cy, w, h = boxes[k].tolist()
                b = [round(min(max(v, 0), 1) * 1000) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]
                if q == "hand" and any(o["label"] == "hand" and iou(o["box"], b) > 0.5 for o in objs):
                    continue
                objs.append({"label": q, "box": b, "score": round(s, 3)})
        objs.sort(key=lambda o: -o["score"])
        return objs[:MAX_OBJECTS]


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1]) / 1e6


def iou(a, b):
    i = max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0) / 1e6
    return i / max(area(a) + area(b) - i, 1e-9)


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
    p.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--box_threshold", type=float, default=0.3)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    # the vision tower's SDPA mask spans every patch in the batch: 12 native-res frames need a 16 GB mask
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="first N rows of each data dir (smoke test)")
    args = p.parse_args()

    lister, boxer = Lister(args.qwen_dir), Boxer(args.gdino, args.box_threshold)
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
        n_obj, fails, t0 = 0, 0, time.time()
        for i in range(0, len(todo), args.batch_size):
            batch, imgs = [], []
            for r in todo[i:i + args.batch_size]:
                try:
                    imgs.append(last_frame(os.path.join(d, r["input_video"]),
                                           os.path.join(gdir, f"{r['sample_id']}_last.jpg")))
                    batch.append(r)
                    fails = 0
                except Exception as e:
                    fails += 1
                    log(f"WARNING: {r['sample_id']}: last frame failed ({str(e)[-200:]})")
                    if fails >= 20:
                        raise RuntimeError("20 consecutive frame-extraction failures") from e
            if not batch:
                continue
            lists = lister(imgs, [r["instruction"] for r in batch])
            with open(path, "a") as f:
                for r, img, text in zip(batch, imgs, lists):
                    objs = boxer(img, parse_names(text))
                    zb = zoom_box(objs)
                    save_zoom(img, zb, os.path.join(gdir, f"{r['sample_id']}_zoom.jpg"))
                    n_obj += len(objs)
                    f.write(json.dumps({"sample_id": r["sample_id"], "frame": f"ground/{r['sample_id']}_last.jpg",
                                        "zoom": f"ground/{r['sample_id']}_zoom.jpg", "zoom_box": zb,
                                        "size": list(img.size), "objects": objs, "names_raw": text}) + "\n")
            k = min(i + args.batch_size, len(todo))
            log(f"{os.path.basename(d)} shard {args.shard}: {k}/{len(todo)}, {n_obj / k:.1f} objects/frame, "
                f"{(time.time() - t0) / k:.2f} s/row | e.g. {[o['label'] for o in objs][:6]}")
    log("GROUNDING DONE")


if __name__ == "__main__":
    main()
