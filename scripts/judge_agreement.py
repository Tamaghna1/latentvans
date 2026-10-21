"""Agreement between the event judge's p_next and human labels (plan.md P27 validation).

Inputs: items.json from make_label_set.py (item -> sample_id, arm), the human labels (a JSON object
{item: "yes"|"no"|"unsure"}, or a directory of <item>.json documents with a "label" field, as the label page's store
exports them), and the judge's per-sample files (<results_dir>/<arm>/judge_<tag>_shard*.jsonl).
"unsure" items are left out of every statistic and counted.

Reports, over the labelled items:
  AUC        P(p_next of a human-yes item > p_next of a human-no item), ties count half
  acc@0.5    judge says yes iff p_next > 0.5
  kappa      Cohen's kappa at 0.5
  per arm    human yes rate vs judge mean p_next and yes rate (does the judge rank the arms as people do?)
with 95% bootstrap intervals over items (seed 0).

  python judge_agreement.py --items items.json --labels labels.json --results_dir results/vans_eval
"""
import argparse
import glob
import json
import os
import random


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def kappa(h, j):
    n = len(h)
    po = sum(a == b for a, b in zip(h, j)) / n
    ph, pj = sum(h) / n, sum(j) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def stats(rows, thr):
    h = [r["human"] for r in rows]
    j = [r["p_next"] > thr for r in rows]
    return {"auc": auc([r["p_next"] for r in rows if r["human"]], [r["p_next"] for r in rows if not r["human"]]),
            "acc": sum(a == b for a, b in zip(h, j)) / len(rows), "kappa": kappa(h, j)}


def load_labels(path):
    if os.path.isdir(path):
        out = {}
        for f in glob.glob(os.path.join(path, "**", "*.json"), recursive=True):
            d = json.load(open(f))
            d = d.get("data", d)
            if "label" in d:
                out[os.path.splitext(os.path.basename(f))[0]] = d["label"]
        return out
    return json.load(open(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--tag", default="Qwen2.5-VL-3B-Instruct")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out", default=None, help="write the report as JSON here")
    args = p.parse_args()

    items = json.load(open(args.items))
    labels = load_labels(args.labels)
    judge = {}
    for arm in {it["arm"] for it in items}:
        for f in glob.glob(os.path.join(args.results_dir, arm, f"judge_{args.tag}_shard*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                judge[(arm, r["sample_id"])] = r["p_next"]
    rows, unsure, missing = [], 0, 0
    for it in items:
        lab = labels.get(it["item"])
        if lab is None:
            missing += 1
            continue
        if lab == "unsure":
            unsure += 1
            continue
        rows.append({**it, "human": lab == "yes", "p_next": judge[(it["arm"], it["sample_id"])]})
    print(f"{len(rows)} labelled yes/no, {unsure} unsure, {missing} not labelled (of {len(items)})")
    if not rows:
        return

    point = stats(rows, args.thr)
    rng = random.Random(0)
    boots = [stats([rng.choice(rows) for _ in rows], args.thr) for _ in range(args.n_boot)]
    report = {"n": len(rows), "unsure": unsure, "missing": missing, "human_yes": sum(r["human"] for r in rows) / len(rows)}
    for k, v in point.items():
        bs = sorted(b[k] for b in boots if b[k] == b[k])
        report[k] = [v, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]]
        print(f"{k:6s} {v:.3f} [{report[k][1]:.3f}, {report[k][2]:.3f}]")
    print(f"human yes rate {report['human_yes']:.3f}")
    report["arms"] = {}
    print(f"{'arm':15s} {'n':>3s} {'human_yes':>9s} {'judge_p':>8s} {'judge_yes':>9s} {'agree':>6s}")
    for arm in sorted({r["arm"] for r in rows}, key=lambda a: -sum(r["human"] for r in rows if r["arm"] == a)):
        rs = [r for r in rows if r["arm"] == arm]
        a = {"n": len(rs), "human_yes": sum(r["human"] for r in rs) / len(rs),
             "judge_p": sum(r["p_next"] for r in rs) / len(rs),
             "judge_yes": sum(r["p_next"] > args.thr for r in rs) / len(rs),
             "agree": sum(r["human"] == (r["p_next"] > args.thr) for r in rs) / len(rs)}
        report["arms"][arm] = a
        print(f"{arm:15s} {a['n']:3d} {a['human_yes']:9.3f} {a['judge_p']:8.3f} {a['judge_yes']:9.3f} {a['agree']:6.3f}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
