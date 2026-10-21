"""Pick the human-label set that validates the event judge (plan.md P27): answer videos from several arms, one
question per item, the judge's own p_next question ("does this video show <next step>?").

Each item is a distinct benchmark sample (nobody sees the same step twice), assigned to one arm; per arm half
COIN and half YouCook; items are shuffled so the arm cannot be guessed from the order. Deterministic (--seed).

  python make_label_set.py --bench_dir data/vans_eval --out items.json
Output: items.json, a list of {item, sample_id, source, arm, step, video}; `video` is relative to the project root
(the real target clip for 'target_real', results/vans_eval/<arm>/videos/<sample_id>.mp4 otherwise).
"""
import argparse
import json
import os
import random

ARMS = ["target_real", "vans_oracle", "vans", "captions_nocot", "copy_input", "copy_last"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per_arm", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    rng = random.Random(args.seed)
    pools = {s: [r for r in rows if r["source"] == s] for s in ("coin", "youcook")}
    for pool in pools.values():
        rng.shuffle(pool)
    half = args.per_arm // 2
    items = []
    for k, arm in enumerate(ARMS):
        for s, pool in pools.items():
            for r in pool[k * half:(k + 1) * half]:
                video = (os.path.join("data/vans_eval", r["target_video"]) if arm == "target_real"
                         else f"results/vans_eval/{arm}/videos/{r['sample_id']}.mp4")
                items.append({"sample_id": r["sample_id"], "source": s, "arm": arm, "step": r["target_sentence"],
                              "video": video})
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item"] = f"i{i + 1:03d}"
    assert len({it["sample_id"] for it in items}) == len(items)
    json.dump(items, open(args.out, "w"), indent=1)
    print(f"{len(items)} items, {len(ARMS)} arms x {args.per_arm} -> {args.out}")


if __name__ == "__main__":
    main()
