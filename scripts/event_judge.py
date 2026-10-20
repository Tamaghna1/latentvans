"""Event-centric judge for next-event answer videos (plan.md P27): does the answer video show the true next step,
and does it stay in the input's scene? CLIP-V/CLIP-T reward reproducing the scene (experiment 9: a frozen last
frame beats every model), so this asks a VLM about the event itself.

Per sample, three yes/no questions to Qwen2.5-VL-Instruct, scored as P(yes) = softmax over the "Yes"/"No" logits
of the first answer token:
  p_next     video only: does it show the benchmark's next-step sentence (target_sentence)
  p_current  video only: does it show the current step (input_sentence); a video that only continues the input
             scores high here, so p_next - p_current separates "did the next step" from "kept doing this one"
  p_scene    last input frame + video: same place and main objects, whatever the action
Answer videos are sampled at --n_frames evenly spaced frames, resized to cover 352x640 like the generators' inputs.

  python event_judge.py --bench_dir data/vans_eval --arm_dir results/vans_eval/vans --qwen_dir <Qwen2.5-VL-7B-Instruct>
  --arm_dir target_real judges the real target clips (upper bound and sanity check).
Output: <arm_dir>/judge_<tag>.jsonl (per sample; resumable, shardable) and, after the last shard, one summary row
in <bench results>/judge_summary.jsonl.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from vans_eval_generate import load_frames  # noqa: E402

NEXT_Q = 'Does this video show the following action being done: "{step}"? Answer Yes or No.'
SCENE_Q = ("The image is the last frame of a video. Is the following video clip set in the same scene as the image, "
           "with the same place and the same main objects? Ignore what action is shown. Answer Yes or No.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Judge:
    def __init__(self, qwen_dir, max_pixels):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.proc = AutoProcessor.from_pretrained(qwen_dir, max_pixels=max_pixels)
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]

    @torch.no_grad()
    def p_yes(self, question, video, image=None):
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "video"}, {"type": "text", "text": question}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True)
        inp = self.proc(text=[text], images=[image] if image is not None else None, videos=[video],
                        return_tensors="pt").to("cuda")
        logits = self.model(**inp).logits[0, -1].float()
        y, n = torch.logsumexp(logits[self.yes], 0), torch.logsumexp(logits[self.no], 0)
        return torch.sigmoid(y - n).item()


def last_input_frame(bench_dir, results_dir, sid):
    for f in (os.path.join(bench_dir, "ground", f"{sid}_last.jpg"),  # ground_objects.py
              os.path.join(results_dir, "copy_last", "videos", f"{sid}_last.jpg")):  # copy_baselines.py
        if os.path.exists(f):
            return Image.open(f).convert("RGB")
    raise FileNotFoundError(f"no last input frame for {sid}")


def summarize(per):
    out = {}
    for src in (None, "coin", "youcook"):
        rows = [p for p in per if src is None or p["source"] == src]
        if not rows:
            continue
        k = src or "overall"
        out[f"{k}/n"] = len(rows)
        for q in ("p_next", "p_current", "p_scene"):
            out[f"{k}/{q}"] = sum(p[q] for p in rows) / len(rows)
            out[f"{k}/{q}_yes"] = sum(p[q] > 0.5 for p in rows) / len(rows)
        out[f"{k}/next_minus_current"] = out[f"{k}/p_next"] - out[f"{k}/p_current"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--arm_dir", required=True, help="dir with videos/<sample_id>.mp4, or 'target_real'")
    p.add_argument("--results_dir", default=None, help="where copy_last/ and judge_summary.jsonl live "
                                                       "(default: parent of --arm_dir)")
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--tag", default=None, help="judge name in output files (default: basename of --qwen_dir)")
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=360 * 640)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()

    tag = args.tag or os.path.basename(os.path.normpath(args.qwen_dir))
    real = args.arm_dir == "target_real"
    results_dir = args.results_dir or os.path.dirname(os.path.normpath(args.arm_dir))
    arm_dir = os.path.join(results_dir, "target_real") if real else args.arm_dir
    arm = os.path.basename(os.path.normpath(arm_dir))
    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    rows = rows[:args.limit] if args.limit else rows
    mine = rows[args.shard::args.n_shards]
    os.makedirs(arm_dir, exist_ok=True)
    out_path = os.path.join(arm_dir, f"judge_{tag}_shard{args.shard}.jsonl")
    done = {json.loads(l)["sample_id"] for l in open(out_path)} if os.path.exists(out_path) else set()

    judge = Judge(args.qwen_dir, args.max_pixels)
    t0, n = time.time(), 0
    for r in mine:
        sid = r["sample_id"]
        if sid in done:
            continue
        path = os.path.join(args.bench_dir, r["target_video"]) if real else os.path.join(arm_dir, "videos", f"{sid}.mp4")
        if not os.path.exists(path):
            log(f"WARNING: no video for {sid}, skipped")
            continue
        video = load_frames(path, n=args.n_frames)
        rec = {"sample_id": sid, "source": r["source"],
               "p_next": judge.p_yes(NEXT_Q.format(step=r["target_sentence"]), video),
               "p_current": judge.p_yes(NEXT_Q.format(step=r["input_sentence"]), video),
               "p_scene": judge.p_yes(SCENE_Q, video, image=last_input_frame(args.bench_dir, results_dir, sid))}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if n == 1 or n % 25 == 0:
            log(f"[{arm} shard {args.shard}] {n} done ({(time.time() - t0) / n:.1f} s/sample) {rec}")

    per = [json.loads(l) for f in sorted(glob.glob(os.path.join(arm_dir, f"judge_{tag}_shard*.jsonl"))) for l in open(f)]
    if len({x["sample_id"] for x in per}) < len(rows):
        log(f"{len(per)}/{len(rows)} judged so far; summary after the last shard")
        return
    summ = summarize(per)
    log(f"{arm} [{tag}] " + " ".join(f"{k}={v:.3f}" for k, v in summ.items() if k.startswith("overall/")))
    with open(os.path.join(results_dir, "judge_summary.jsonl"), "a") as f:
        f.write(json.dumps({"arm": arm, "judge": tag, "n_frames": args.n_frames, **summ, "time": time.time()}) + "\n")
    from wandb_util import log_summary
    log_summary(args.wandb_project, f"judge-{tag}-{arm}", vars(args), summ, job_type="event_judge")
    log("EVENT JUDGE DONE")


if __name__ == "__main__":
    main()
