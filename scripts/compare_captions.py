"""Paired comparison of caption arms on the VANS benchmark: per-sample ROUGE-L F1 (the benchmark's metric:
lowercased [a-z0-9]+ tokens, LCS F1, as rouge-score's rougeL without stemming), mean difference vs a
baseline arm over the samples both have, and a paired bootstrap 95% CI.

  python compare_captions.py --bench data/vans_eval/benchmark.jsonl \
      --base 'results/vans_coconut/coconut_nocot/captions_shard*.jsonl' \
      --arm ground_text='results/vans_ground/ground_text/captions_shard*.jsonl' ...
Pure Python, no GPU.
"""
import argparse
import glob
import json
import random
import re


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def rouge_l(pred, ref):
    a, b = tok(pred), tok(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def load(pattern, gt):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            c = json.loads(l)
            out[c["sample_id"]] = rouge_l(c["caption"], gt[c["sample_id"]])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--base", required=True, help="glob of the baseline arm's caption jsonl files")
    p.add_argument("--arm", action="append", default=[], help="name=glob, repeatable")
    p.add_argument("--n_boot", type=int, default=5000)
    args = p.parse_args()
    gt = {}
    for l in open(args.bench):
        r = json.loads(l)
        gt[r["sample_id"]] = r["gt_caption"]
    base = load(args.base, gt)
    print(f"base: n={len(base)} ROUGE-L={sum(base.values()) / max(len(base), 1):.4f}")
    rng = random.Random(0)
    for spec in args.arm:
        name, pattern = spec.split("=", 1)
        arm = load(pattern, gt)
        ids = sorted(set(arm) & set(base))
        if not ids:
            print(f"{name}: no samples in common with the baseline")
            continue
        d = [arm[i] - base[i] for i in ids]
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(args.n_boot))
        lo, hi = boots[int(0.025 * args.n_boot)], boots[int(0.975 * args.n_boot)]
        print(f"{name}: n={len(ids)} ROUGE-L={sum(arm[i] for i in ids) / len(ids):.4f} "
              f"diff vs base={sum(d) / len(d):+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 or hi < 0 else ''}")


if __name__ == "__main__":
    main()
