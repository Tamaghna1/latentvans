"""Training-free text baselines for the caption metric (P23). Writes caption jsonl files that
compare_captions.py scores like any arm.

  python text_baselines.py --train data/vans_train/benchmark.jsonl --bench data/vans_eval/benchmark.jsonl \
      --out_dir results/text_baselines
Pure Python, no GPU.

  const        the single training caption with the highest mean ROUGE-L against other training captions
  nn_instr     the caption of the training row whose instruction is most similar (tf-idf cosine)
  nn_target    the caption of the training row whose target step sentence is most similar (oracle: uses
               the test row's ground-truth step text, but no video)
  input_sent / target_sent   the short step sentences themselves (style mismatch reference)
"""
import argparse
import json
import math
import os
import random
from collections import Counter

from compare_captions import rouge_l, tok

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--bench", required=True)
p.add_argument("--out_dir", required=True)
args = p.parse_args()
train = [json.loads(l) for l in open(args.train)]
test = [json.loads(l) for l in open(args.bench)]
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

tv, sv = {r["vid"] for r in train}, {r["vid"] for r in test}
print(f"train rows {len(train)} vids {len(tv)}; test rows {len(test)} vids {len(sv)}; shared vids {len(tv & sv)}")
tt, st = Counter(r["task"] for r in train), {r["task"] for r in test}
print(f"test tasks also in train: {sum(t in tt for t in st)}/{len(st)}")


def write(name, caps):
    with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
        for r, c in zip(test, caps):
            f.write(json.dumps({"sample_id": r["sample_id"], "caption": c}) + "\n")


write("input_sent", [r["input_sentence"] for r in test])
write("target_sent", [r["target_sentence"] for r in test])

rng = random.Random(0)
cands = rng.sample(train, 300)
refs = rng.sample(train, 300)
best = max(cands, key=lambda c: sum(rouge_l(c["gt_caption"], r["gt_caption"]) for r in refs if r is not c))
print("const caption:", best["gt_caption"][:200])
write("const", [best["gt_caption"]] * len(test))


def tfidf(field):
    df = Counter(w for r in train for w in set(tok(r[field])))
    idf = {w: math.log(len(train) / (1 + n)) for w, n in df.items()}

    def vec(s):
        c = Counter(tok(s))
        v = {w: n * idf.get(w, math.log(len(train))) for w, n in c.items()}
        z = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / z for w, x in v.items()}

    tr = [vec(r[field]) for r in train]

    def nn(s):
        q = vec(s)
        return max(range(len(train)), key=lambda i: sum(x * tr[i].get(w, 0.0) for w, x in q.items()))
    return nn


nn = tfidf("instruction")
write("nn_instr", [train[nn(r["instruction"])]["gt_caption"] for r in test])
nn = tfidf("target_sentence")
write("nn_target", [train[nn(r["target_sentence"])]["gt_caption"] for r in test])
print("written to", out_dir)
