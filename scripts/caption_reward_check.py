"""Which text-only caption score tracks the event judge (plan.md P29 reward check)? Spearman between per-sample
caption scores (ROUGE-L to the GT caption, ROUGE-L to the step sentence, share of the step's content words in the
caption) and the judge's p_next of the video rendered from that caption. Args: a dir holding label/bench/benchmark.jsonl,
capsig/<caption files> and judge/<arm>/judge_*.jsonl (copied from the cluster).
"""
import json, glob, re, random, sys
SP = sys.argv[1]
B = {r["sample_id"]: r for r in map(json.loads, open(f"{SP}/label/bench/benchmark.jsonl"))}
def caps(pat): return {c["sample_id"]: c["caption"] for f in glob.glob(pat) for c in map(json.loads, open(f))}
C = {"captions_nocot": caps(f"{SP}/capsig/vans_coconut/coconut_nocot/captions_shard*.jsonl"),
     "vans": caps(f"{SP}/capsig/vans_eval/vans/captions*.jsonl")}
J = {a: {r["sample_id"]: r for r in map(json.loads, open(f"{SP}/judge/{a}/judge_Qwen2.5-VL-3B-Instruct_shard0.jsonl"))} for a in C}
STOP = set("a an the and of to in on with into is are be it its this that for from by at as then while using onto over some".split())
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
def lcs(a, b):
    d = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = d[j + 1]; d[j + 1] = prev + 1 if x == y else max(d[j + 1], d[j]); prev = cur
    return d[-1]
def rougeL(c, r):
    a, b = tok(c), tok(r); l = lcs(a, b)
    return 0 if l == 0 else 2 * (l / len(a)) * (l / len(b)) / ((l / len(a)) + (l / len(b)))
def recall(c, r):  # content words of the step sentence found in the caption
    rc = {w for w in tok(r) if w not in STOP}; cc = set(tok(c))
    return len(rc & cc) / max(1, len(rc))
def rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def spear(x, y):
    rx, ry = rank(x), rank(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
def boot(f, rows, n=1000):
    rng = random.Random(0); v = sorted(f([rng.choice(rows) for _ in rows]) for _ in range(n)); return v[25], v[975]
rows = []
for a in C:
    for sid, c in C[a].items():
        r = B[sid]
        rows.append({"arm": a, "p": J[a][sid]["p_next"], "rouge_gtcap": rougeL(c, r["gt_caption"]),
                     "rouge_step": rougeL(c, r["target_sentence"]), "recall_step": recall(c, r["target_sentence"]),
                     "recall_input": recall(c, r["input_sentence"])})
print(f"n={len(rows)} (nocot + vans captions; p = judge p_next of the rendered video)")
for k in ("rouge_gtcap", "rouge_step", "recall_step", "recall_input"):
    f = lambda rs: spear([r[k] for r in rs], [r["p"] for r in rs])
    lo, hi = boot(f, rows)
    per = "  ".join(f"{a}={spear([r[k] for r in rows if r['arm']==a], [r['p'] for r in rows if r['arm']==a]):+.3f}" for a in C)
    print(f"spearman({k:12s}, p_next) = {f(rows):+.3f} [{lo:+.3f}, {hi:+.3f}]   {per}")
# GT-caption videos: their caption's rouge to itself is 1; how does p_next vary with the step's content-word recall in the GT caption?
o = {r["sample_id"]: r for r in map(json.loads, open(f"{SP}/judge/vans_oracle/judge_Qwen2.5-VL-3B-Instruct_shard0.jsonl"))}
x = [recall(B[s]["gt_caption"], B[s]["target_sentence"]) for s in o]; y = [o[s]["p_next"] for s in o]
print(f"GT captions: mean step-word recall {sum(x)/len(x):.3f}; spearman(recall_step, p_next of GT-caption video) = {spear(x, y):+.3f}")
for a in C:
    rs = [r for r in rows if r["arm"] == a]
    print(f"{a}: mean rouge_gtcap {sum(r['rouge_gtcap'] for r in rs)/len(rs):.3f} recall_step {sum(r['recall_step'] for r in rs)/len(rs):.3f} recall_input {sum(r['recall_input'] for r in rs)/len(rs):.3f}")
