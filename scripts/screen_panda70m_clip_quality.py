#!/usr/bin/env python
"""
Automated, scaled-up quality screening for downloaded Panda-70M clips.

Why this exists
----------------
verify_panda70m_clips.py's own approach (extract a few thumbnails per clip,
a human eyeballs them next to the TwiFF question/answer) is exactly how the
two previous content-mismatch bugs in this project were actually caught --
see claude/latentvans-handoff.md's SLURM gotchas section for the volleyball/
asparagus story. That approach does not scale to 1879 downloaded clips
(the current full-dataset download_panda70m_clips.py run): nobody is going
to watch 1879 clips by hand. Manual spot-checking a small random sample
(done by hand on 9 clips before this script existed: 8/9 correct) gives only
a rough, low-confidence estimate of the true mismatch rate across the whole
1879, and can't tell you WHICH of the 1879 are the bad ones.

What this script does instead: uses Qwen2.5-VL-3B-Instruct -- already
staged on this cluster for Stage 0 training -- as a zero-shot judge. Pure
inference, no training, no gradient, no LoRA. For every downloaded clip, it
samples a few frames, shows them to the VLM next to that clip's TwiFF
question/answer text, and asks the VLM to say whether the frames plausibly
match. This scales to the full 1879 (or however many clips exist), and
turns "check a handful of clips and hope they're representative" into
"every clip gets screened, and the model tells you which ones are worth a
human's actual attention."

IMPORTANT -- this is a screening tool, not ground truth
---------------------------------------------------------
Qwen2.5-VL-3B-Instruct is a 3B model doing a zero-shot judgment call on a
single frame triplet -- it will get things wrong in both directions (miss
real mismatches, and flag some correct clips it's unsure about). Treat its
verdict as a triage signal, not a final answer. The recommended workflow:
  1. Run this script across all downloaded clips (see Usage).
  2. Read quality_screen_flagged.json -- every clip the VLM called MISMATCH
     or that it failed to give a parseable verdict for. Manually eyeball a
     good chunk of these (the frames it judged from are saved right next to
     the report, so no separate thumbnail-extraction step is needed) --
     this is a much smaller, much more useful list than 1879 random clips.
  3. ALSO manually eyeball a small random sample of clips the VLM called
     MATCH (10-20), the same way clips were checked before this script
     existed. If the VLM is missing real mismatches, this is how you'd find
     out -- don't just trust it because it said MATCH.
  4. Only after both (2) and (3) look reasonable, treat the surviving
     MATCH set as good enough to train Stage 0 on.

What "plausibly match" means here: does the sampled visual content look
consistent with the question/answer text, the same visual-plausibility
judgment a human was making by eye in the earlier manual checks -- not
"is this a great QA pair for training" or "is the question well-written."

Prerequisites
-------------
- Qwen2.5-VL-3B-Instruct already staged (same --qwen_dir as
  train_stage0_latent_grounding.py) -- data_staging.sh already does this.
- Clips already downloaded via download_panda70m_clips.py.
- System `ffmpeg`/`ffprobe` on PATH (same as verify_panda70m_clips.py --
  download_panda70m_clips.sh already installs this).
- transformers + torch already installed in this env (train_stage0's own
  dependency list) -- this script deliberately adds NO new pip
  dependencies (no decord, no peft -- this is read-only inference, not
  training) to avoid any risk of an --upgrade line disturbing the pinned
  torch stack (see claude/latentvans-handoff.md's ncclCommResume saga).

Usage
-----
    python screen_panda70m_clip_quality.py \\
        --qwen_dir /scratch/users/anirban/tamaghnam/latentvans/checkpoints/Qwen2.5-VL-3B-Instruct \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/twiff_metadata_2000 \\
        --clips_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips

Safe to resubmit / test on a subset first: results are appended to
--report_path (default <clips_dir>/quality_screen_report.jsonl) one line
per clip as they're judged, and clips already present in that file are
skipped on a re-run -- same "safe to resubmit, already-done work isn't
redone" pattern as download_panda70m_clips.py. Pass --limit N to try this
on a small batch first before committing a multi-hour GPU job to all 1879.

Output
------
- <clips_dir>/quality_screen_report.jsonl: one JSON object per judged clip
  (video, verdict, reason, raw_model_output, frame_paths, question, answer).
- <clips_dir>/quality_screen_flagged.json: just the MISMATCH/PARSE_ERROR/
  no-metadata entries, for fast manual review (step 2 above).
- <clips_dir>/_quality_screen_frames/: the actual frames each clip was
  judged from, so a flagged clip can be eyeballed immediately without
  re-extracting anything.

Known limitations (read before trusting the MATCH bucket blindly)
---------------------------------------------------------------------
1. Only a handful of frames per clip (--num_frames, default 3) are shown --
   same tradeoff verify_panda70m_clips.py made for human thumbnails. A
   short action that falls between sampled frames could be missed by the
   VLM just as it could be missed by a human skimming 3 thumbnails.
2. Like verify_panda70m_clips.py, if a clip file has more than one TwiFF
   row (multiple questions about the same clip), only the first row found
   is used for the judgment -- consistent with that script's existing
   choice, not a new limitation introduced here.
3. The VLM has not itself been validated as a good judge for this exact
   task -- that's what workflow step 3 above is for. Don't skip it just
   because this script makes screening 1879 clips feel automated.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback

import torch


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dtype_from_str(s):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def ffmpeg_available():
    import shutil
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def ffprobe_duration(path):
    """Returns duration in seconds, or None on failure. Same approach as
    verify_panda70m_clips.py's ffprobe_info(), duration-only."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        return None


def extract_sample_frames(clip_path, duration, out_dir, stem, num_frames):
    """Evenly-spaced frames (avoiding the very first/last, often black/transition
    frames) -- identical spacing logic to verify_panda70m_clips.py's
    extract_thumbnails(), so a clip's screened frames match what a human would
    see if they ran that script on the same clip. Returns list of written paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    if not duration or duration <= 0:
        return written
    fractions = [(i + 1) / (num_frames + 1) for i in range(num_frames)]
    for i, frac in enumerate(fractions):
        ts = duration * frac
        out_path = os.path.join(out_dir, f"{stem}_frame{i}.jpg")
        cmd = ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", clip_path, "-frames:v", "1", "-q:v", "2", out_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(out_path):
            written.append(out_path)
    return written


def load_twiff_by_video(metadata_dir):
    """video_filename -> {question, answer} for every staged TwiFF row.
    First row wins if a filename has multiple QA rows -- same choice
    verify_panda70m_clips.py already made (see module docstring, limitation 2)."""
    from datasets import load_from_disk
    rows = load_from_disk(metadata_dir)
    by_video = {}
    for row in rows:
        by_video.setdefault(row["video"], {"question": row["question"], "answer": row["answer"]})
    return by_video


def build_judge_messages(frames, question, answer, max_answer_chars):
    answer_trunc = answer[:max_answer_chars] if answer else ""
    prompt_text = (
        f"Question written about this clip: {question}\n\n"
        f"Answer written about this clip: {answer_trunc}\n\n"
        "Do the frames above plausibly show the scene described by this question and "
        "answer (same setting, people/objects, and action)? Judge visual content only -- "
        "not whether the question is well-phrased. Respond in EXACTLY this format, nothing else:\n"
        "VERDICT: MATCH or MISMATCH\n"
        "REASON: <one short sentence>"
    )
    return [
        {"role": "system", "content": (
            "You are a careful data-quality auditor for a video question-answering dataset. "
            "You will see a few frames sampled from a short video clip, followed by a question "
            "and an answer an annotator wrote about that exact clip. Your only job is to judge "
            "whether the frames are visually consistent with that question and answer."
        )},
        {"role": "user", "content": (
            [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt_text}]
        )},
    ]


def parse_verdict(raw_text):
    upper = raw_text.upper()
    if "MISMATCH" in upper:
        verdict = "MISMATCH"
    elif "MATCH" in upper:
        verdict = "MATCH"
    else:
        verdict = "PARSE_ERROR"
    m = re.search(r"REASON:\s*(.+)", raw_text, re.IGNORECASE)
    reason = m.group(1).strip() if m else raw_text.strip()
    return verdict, reason


def judge_clip(vlm, processor, tokenizer, frames, question, answer, device, max_answer_chars, max_new_tokens):
    from PIL import Image

    pil_frames = [Image.open(p).convert("RGB") for p in frames]
    messages = build_judge_messages(pil_frames, question, answer, max_answer_chars)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=pil_frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    raw_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
    verdict, reason = parse_verdict(raw_text)
    return verdict, reason, raw_text


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qwen_dir", required=True, help="Same Qwen2.5-VL-3B-Instruct dir train_stage0 uses")
    p.add_argument("--metadata_dir", required=True, help="Staged TwiFF-2.7M slice (save_to_disk dir)")
    p.add_argument("--clips_dir", required=True, help="download_panda70m_clips.py's --output_dir")
    p.add_argument("--report_path", default=None,
                   help="JSONL, one line per judged clip (default: <clips_dir>/quality_screen_report.jsonl)")
    p.add_argument("--flagged_path", default=None,
                   help="JSON summary of just the flagged clips (default: <clips_dir>/quality_screen_flagged.json)")
    p.add_argument("--frames_dir", default=None,
                   help="Where to save the frames each clip was judged from "
                        "(default: <clips_dir>/_quality_screen_frames)")
    p.add_argument("--num_frames", type=int, default=3, help="Evenly-spaced frames per clip shown to the VLM")
    p.add_argument("--max_answer_chars", type=int, default=600,
                   help="Truncate TwiFF answer text before it goes in the judge prompt")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--limit", type=int, default=None, help="Only judge this many NOT-yet-judged clips (for testing)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    report_path = args.report_path or os.path.join(args.clips_dir, "quality_screen_report.jsonl")
    flagged_path = args.flagged_path or os.path.join(args.clips_dir, "quality_screen_flagged.json")
    frames_dir = args.frames_dir or os.path.join(args.clips_dir, "_quality_screen_frames")

    if not ffmpeg_available():
        log("FATAL: `ffmpeg`/`ffprobe` not found on PATH.")
        sys.exit(1)

    cuda_ok = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_ok:
        log("FATAL: --device cuda requested but CUDA is not available.")
        sys.exit(1)
    # Same reasoning as train_stage0_latent_grounding.py: a GPU can be allocated by
    # SLURM while torch still can't see it (torch/driver CUDA mismatch -- see
    # claude/latentvans-handoff.md, item 9). Judging ~1879 clips one-by-one on CPU
    # with a 3B VLM would silently turn a ~few-hour GPU job into a multi-day one --
    # refuse instead of quietly doing that.
    gpu_was_allocated = bool(
        os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if args.device == "auto" and not cuda_ok and gpu_was_allocated:
        log("FATAL: this job was allocated a GPU but torch.cuda.is_available() is False -- almost certainly "
            "the torch/CUDA driver mismatch documented in claude/latentvans-handoff.md. Refusing to silently "
            "run VLM inference over ~thousands of clips on CPU. Fix the torch install and resubmit, or pass "
            "--device cpu explicitly if you actually want a (very slow) CPU run.")
        sys.exit(1)

    use_cuda = cuda_ok and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    compute_dtype = dtype_from_str(args.dtype) if use_cuda else torch.float32
    log(f"torch={torch.__version__} device={device} compute_dtype={compute_dtype}")

    log("=" * 70)
    log(f"Loading Qwen2.5-VL-3B-Instruct from {args.qwen_dir} (inference only, no training)")
    log("=" * 70)
    processor = AutoProcessor.from_pretrained(args.qwen_dir)
    tokenizer = processor.tokenizer
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_dir, dtype=compute_dtype, device_map=None, low_cpu_mem_usage=True,
    ).to(device)
    vlm.eval()

    log(f"Loading TwiFF metadata from {args.metadata_dir} ...")
    by_video = load_twiff_by_video(args.metadata_dir)

    clip_files = sorted(
        f for f in os.listdir(args.clips_dir)
        if f.endswith(".mp4") and os.path.isfile(os.path.join(args.clips_dir, f))
    )
    log(f"Found {len(clip_files)} downloaded clip(s) in {args.clips_dir}")

    already_done = set()
    if os.path.exists(report_path):
        with open(report_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    already_done.add(json.loads(line)["video"])
                except (json.JSONDecodeError, KeyError):
                    pass
        log(f"{len(already_done)} clip(s) already judged in a prior run of {report_path} -- skipping those")

    pending = [f for f in clip_files if f not in already_done]
    if args.limit is not None:
        pending = pending[:args.limit]
    log(f"{len(pending)} clip(s) to judge this run")

    counts = {"MATCH": 0, "MISMATCH": 0, "PARSE_ERROR": 0, "NO_METADATA": 0}
    report_f = open(report_path, "a")
    try:
        for idx, fname in enumerate(pending):
            clip_path = os.path.join(args.clips_dir, fname)
            stem = fname[:-4]
            twiff_row = by_video.get(fname)

            if twiff_row is None:
                counts["NO_METADATA"] += 1
                entry = {"video": fname, "verdict": "NO_METADATA", "reason":
                         "filename not found in staged TwiFF metadata -- can't judge without a question/answer",
                         "raw_model_output": None, "frame_paths": [], "question": None, "answer": None}
                report_f.write(json.dumps(entry) + "\n")
                report_f.flush()
                continue

            duration = ffprobe_duration(clip_path)
            frames = extract_sample_frames(clip_path, duration, frames_dir, stem, args.num_frames)
            if not frames:
                counts["PARSE_ERROR"] += 1
                entry = {"video": fname, "verdict": "PARSE_ERROR", "reason":
                         "no frames could be extracted (missing/zero duration or ffmpeg failure)",
                         "raw_model_output": None, "frame_paths": [],
                         "question": twiff_row["question"], "answer": twiff_row["answer"]}
                report_f.write(json.dumps(entry) + "\n")
                report_f.flush()
                continue

            try:
                verdict, reason, raw_text = judge_clip(
                    vlm, processor, tokenizer, frames, twiff_row["question"], twiff_row["answer"],
                    device, args.max_answer_chars, args.max_new_tokens,
                )
            except Exception as e:
                verdict, reason, raw_text = "PARSE_ERROR", f"exception during judging: {e}", None

            counts[verdict] = counts.get(verdict, 0) + 1
            entry = {
                "video": fname, "verdict": verdict, "reason": reason, "raw_model_output": raw_text,
                "frame_paths": frames, "question": twiff_row["question"], "answer": twiff_row["answer"],
            }
            report_f.write(json.dumps(entry) + "\n")
            report_f.flush()

            if (idx + 1) % args.log_every == 0 or (idx + 1) == len(pending):
                log(f"  [{idx + 1}/{len(pending)}] {fname}: {verdict} -- {reason[:100] if reason else ''} "
                    f"(running: {counts})")
    finally:
        report_f.close()

    # Rebuild the flagged summary from the FULL report (not just this run's increment) --
    # so re-running with --limit multiple times still produces a complete, up-to-date
    # flagged list each time, not just whatever was flagged in the most recent invocation.
    all_entries = []
    with open(report_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_entries.append(json.loads(line))
    flagged = [e for e in all_entries if e["verdict"] in ("MISMATCH", "PARSE_ERROR", "NO_METADATA")]
    with open(flagged_path, "w") as f:
        json.dump(flagged, f, indent=2)

    total = len(all_entries)
    n_match = sum(1 for e in all_entries if e["verdict"] == "MATCH")
    log("=" * 70)
    log(f"DONE this run. {len(pending)} clip(s) judged just now.")
    log(f"Cumulative across {report_path}: {total} clip(s) judged total, "
        f"{n_match} MATCH ({100 * n_match / total:.1f}%)" if total else "no clips judged yet")
    log(f"{len(flagged)} clip(s) flagged for manual review -> {flagged_path}")
    log("=" * 70)
    log("NEXT: read quality_screen_flagged.json and manually eyeball a good chunk of it (frames are "
        "already saved under _quality_screen_frames/, no re-extraction needed). ALSO manually eyeball "
        "a small random sample of MATCH-verdict clips -- this script's judgment is a screening heuristic "
        "from a 3B model, not ground truth. See this script's module docstring for the full recommended "
        "workflow.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL: unhandled exception.")
        traceback.print_exc()
        sys.exit(1)
