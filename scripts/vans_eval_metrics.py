"""Score generated VANS-benchmark answers with the paper's metrics (arXiv 2511.16669, Table 1).

Text (arms with a predicted caption): BLEU@1-4 (corpus-level, regex word tokens, lowercased) and
ROUGE-L F1 (rouge-score, mean over samples) against the ground-truth caption.

Video, against the ground-truth target clip, both sides as 33 evenly spaced frames at 352x640
(resize-to-cover + center crop; generated videos with a different frame count are resampled to
33 by nearest index):
  CLIP-V  ViT-B/32 cosine between generated frame i and GT frame i, averaged over frames, then samples
  CLIP-T  ViT-B/32 cosine between each generated frame and the GT caption, averaged
  FVD     StyleGAN-V I3D features of 16 evenly spaced frames at 224x224, over all samples

The paper does not say what CLIP-T's text is; we use the GT caption (does the video show the right
next event). Metrics are reported overall and per source (youcook / coin).

Output: <arm_dir>/metrics.json, and one row per arm appended to <out_dir>/summary.jsonl.
"""
import argparse
import glob
import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
H, W, NF = 352, 640, 33


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_video(path, n=NF):
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out = []
    for i in idx:
        im = Image.fromarray(frames[i])
        s = max(H / im.height, W / im.width)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        l, t = (im.width - W) // 2, (im.height - H) // 2
        out.append(np.asarray(im.crop((l, t, l + W, t + H))))
    return np.stack(out)  # (n, H, W, 3) uint8


# ---------------- text ----------------
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def corpus_bleu(hyps, refs, max_n=4):
    """Papineni corpus BLEU with one reference; returns BLEU@1..max_n."""
    clipped = [0] * max_n
    total = [0] * max_n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            clipped[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            total[n - 1] += max(len(h) - n + 1, 0)
    bp = 1.0 if hyp_len > ref_len else float(np.exp(1 - ref_len / max(hyp_len, 1)))
    out = []
    for n in range(1, max_n + 1):
        ps = [clipped[k] / total[k] if total[k] else 0.0 for k in range(n)]
        out.append(bp * float(np.exp(np.mean(np.log(ps)))) if min(ps) > 0 else 0.0)
    return out


def text_metrics(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"])
    rl = [sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    b = corpus_bleu([tok(p) for p in preds], [tok(r) for r in refs])
    return {**{f"BLEU@{i + 1}": v for i, v in enumerate(b)}, "ROUGE-L": float(np.mean(rl))}


# ---------------- video ----------------
class Clip:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

    @torch.no_grad()
    def images(self, frames):
        x = self.p(images=[Image.fromarray(f) for f in frames], return_tensors="pt")["pixel_values"]
        return F.normalize(self.m.get_image_features(pixel_values=x.to(self.device)), dim=-1)

    @torch.no_grad()
    def text(self, s):
        x = self.p(text=[s], return_tensors="pt", padding=True, truncation=True, max_length=77)
        return F.normalize(self.m.get_text_features(**{k: v.to(self.device) for k, v in x.items()}), dim=-1)


class I3D:
    def __init__(self, path, device):
        if not os.path.exists(path):
            torch.hub.download_url_to_file(I3D_URL, path)
        self.m = torch.jit.load(path).eval().to(device)
        self.device = device

    @torch.no_grad()
    def features(self, frames):
        idx = np.linspace(0, len(frames) - 1, 16).round().astype(int)
        x = torch.from_numpy(frames[idx]).permute(3, 0, 1, 2).float()  # (3, 16, H, W)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x / 127.5 - 1).unsqueeze(0).to(self.device)
        return self.m(x, rescale=False, resize=False, return_features=True).cpu().numpy()[0]


def frechet(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--arm_dir", required=True, help="dir with videos/<sample_id>.mp4 and optional captions*.jsonl")
    p.add_argument("--i3d", required=True, help="path to (or where to download) i3d_torchscript.pt")
    p.add_argument("--summary", default=None, help="jsonl to append this arm's overall row to")
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = os.path.basename(os.path.normpath(args.arm_dir))

    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    caps = {}
    for f in glob.glob(os.path.join(args.arm_dir, "captions*.jsonl")):
        for l in open(f):
            c = json.loads(l)
            caps[c["sample_id"]] = c["caption"]
    rows = [r for r in rows if os.path.exists(os.path.join(args.arm_dir, "videos", f"{r['sample_id']}.mp4"))]
    log(f"{arm}: {len(rows)} generated samples, {len(caps)} captions")

    clip, i3d = Clip(device), I3D(args.i3d, device)
    per = []
    feats_gen, feats_gt = [], []
    for k, r in enumerate(rows):
        gen = read_video(os.path.join(args.arm_dir, "videos", f"{r['sample_id']}.mp4"))
        gt = read_video(os.path.join(args.bench_dir, r["target_video"]))
        eg, et = clip.images(gen), clip.images(gt)
        tt = clip.text(r["gt_caption"])
        per.append({"sample_id": r["sample_id"], "source": r["source"],
                    "clip_v": float((eg * et).sum(-1).mean()), "clip_t": float((eg @ tt.T).mean())})
        feats_gen.append(i3d.features(gen))
        feats_gt.append(i3d.features(gt))
        if (k + 1) % 50 == 0:
            log(f"  {k + 1}/{len(rows)}")
    feats_gen, feats_gt = np.stack(feats_gen), np.stack(feats_gt)

    def summarize(mask):
        sel = [x for x, m in zip(per, mask) if m]
        idx = [i for i, m in enumerate(mask) if m]
        out = {"n": len(sel), "FVD": frechet(feats_gen[idx], feats_gt[idx]),
               "CLIP-V": float(np.mean([x["clip_v"] for x in sel])),
               "CLIP-T": float(np.mean([x["clip_t"] for x in sel]))}
        ids = [x["sample_id"] for x in sel]
        gt = {r["sample_id"]: r["gt_caption"] for r in rows}
        # text metrics only for predicted captions (not for oracle arms, whose caption is the GT itself)
        if all(i in caps for i in ids) and any(caps[i].strip() != gt[i].strip() for i in ids):
            out.update(text_metrics([caps[i] for i in ids], [gt[i] for i in ids]))
        return out

    res = {"arm": arm, "overall": summarize([True] * len(per))}
    for src in sorted({x["source"] for x in per}):
        res[src] = summarize([x["source"] == src for x in per])
    json.dump({**res, "per_sample": per}, open(os.path.join(args.arm_dir, "metrics.json"), "w"), indent=1)
    for k in ["overall"] + sorted({x["source"] for x in per}):
        log(f"{arm} [{k}] " + " ".join(f"{m}={v:.4f}" if isinstance(v, float) else f"{m}={v}" for m, v in res[k].items()))
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(json.dumps({"arm": arm, **{f"{k}/{m}": v for k in res if k != "arm" for m, v in res[k].items()},
                                "time": time.time()}) + "\n")
    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name=f"vans-eval-{arm}", config={"arm": arm})
            run.log({f"{k}/{m}": v for k in res if k != "arm" for m, v in res[k].items()})
            run.finish()
        except Exception as e:
            log(f"WARNING: W&B logging failed ({e})")


if __name__ == "__main__":
    main()
