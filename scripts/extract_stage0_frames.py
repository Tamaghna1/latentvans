#!/usr/bin/env python
"""
Frame extraction for LatentVANS Stage 0 -- precomputes the exact frames
train_stage0_latent_grounding.py needs (context + future/reasoning frames)
from the already-downloaded Panda-70M clips, ONCE, instead of leaving every
single training draw to live-decode the full clip via decord.

Why this exists (2026-09-24)
------------------------------------------------------------------------
train_stage0_latent_grounding.py's TwiFFStage0Stream.next_example() calls
load_video_frames(video_path, indices) TWICE per training example (once
for row["question_images_index"], once for row["reasoning_images_index"]),
each time opening the FULL clip via decord.VideoReader(...) and seeking to
a handful of specific frame indices. Two problems with doing that live,
every step, for the life of a multi-hour run:

1. THE DECORD DECODE-FAILURE RATE IS REAL AND CONSTANT, NOT A ONE-TIME
   COST. claude/latentvans-handoff.md item 12 documents a sustained
   ~38-41% decord decode-failure rate across job 57910's full 2000-step
   run (video_reader.cc "cannot find video stream" / threaded_decoder.cc
   avcodec_send_packet EAGAIN), and this project's 2026-09-24 analysis of
   job 61184 found the same ~30-34% rate held roughly constant across that
   run too. A clip that fails to decode fails EVERY time it's drawn, for
   the entire run -- TwiFFStage0Stream.next_example() just catches the
   exception, logs "skipping <video>: <e>", and tries another random row.
   That means the exact same broken clips get re-opened, re-attempted, and
   re-logged over and over across a run's full 16,000+ example draws, at
   real wall-clock cost, instead of the failure being discovered once.
2. DISK FOOTPRINT. A full downloaded clip (several seconds up to ~48s of
   video, per download_panda70m_clips.py) is far larger than the 1-8
   individual frames train_stage0_latent_grounding.py actually reads from
   it (TwiFF/Future-L1-50K's reasoning_images_index is documented at 1-7
   frames; question_images_index is typically similarly small). This
   project's cluster account has now hit OSError: Disk quota exceeded
   TWICE (once traced to `_raw_downloads/` staging leftovers, once fatal
   mid-training-run on 2026-09-24) -- keeping every full clip around when
   only a few frames per clip are ever read is a real, avoidable
   contributor to that pressure.

WHAT THIS SCRIPT DOES NOT DO -- READ BEFORE RUNNING
------------------------------------------------------------------------
This does NOT reduce the underlying decord decode-failure rate. A clip
decord cannot decode here will not decode any better here than it does
inside train_stage0_latent_grounding.py, because this calls decord exactly
the same way (same decord.bridge.set_bridge("native") / VideoReader /
get_batch call pattern as load_video_frames). It only means that failure
gets discovered and recorded ONCE (in extract_report.jsonl /
extract_summary.json) instead of silently re-costing every training step
for the life of a run. If you want the failure rate ITSELF to go down,
that is claude/latentvans-handoff.md item 12's still-unapplied fix (a
decode-validation pass inside download_panda70m_clips.py's cut_clip(),
applied via a --force_recut pass) -- run that FIRST if you want it, then
re-run this extraction script against the repaired clips. Doing it in the
other order just means re-extracting the same clips a second time.

THIS SCRIPT ALSO DOES NOT, BY ITSELF, CHANGE
train_stage0_latent_grounding.py'S BEHAVIOR. TwiFFStage0Stream /
load_video_frames still live-decode raw video from --video_root exactly as
before -- this only builds a frame cache alongside it. Wiring the training
script to read pre-extracted frames from this cache instead of the raw
clips (which is what would actually remove decord and the raw clips from
the training-time critical path) is a separate change, not made here.

What it extracts
------------------------------------------------------------------------
For every row in --metadata_dir (same TwiFF/Future-L1-50K schema
train_stage0_latent_grounding.py already requires: video, question,
answer, question_images_index, reasoning_images_index, meta_data), this
groups rows by their `video` filename (a clip can be referenced by more
than one QA row -- same dedup-by-filename pattern download_panda70m_clips.py
already uses for source videos) and takes the UNION of that video's
question_images_index and reasoning_images_index across every row that
references it -- the full set of frame indices any training example could
ever ask that clip for. Each frame is decoded once via decord and saved as
a JPEG at --output_dir/<video_stem>/<frame_idx>.jpg.

Usage
-----
    python extract_stage0_frames.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/future_l1_50k_metadata_full \\
        --video_root /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_frames \\
        --num_workers 8

Needs `decord` (already a training dependency) and Pillow.
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time
import traceback


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata_dir", required=True,
                   help="Path to the staged TwiFF/Future-L1-50K metadata slice (save_to_disk dir) -- "
                        "same schema train_stage0_latent_grounding.py requires.")
    p.add_argument("--video_root", required=True,
                   help="Directory of already-downloaded Panda-70M clips (download_panda70m_clips.py's "
                        "--output_dir) to read frames from.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write extracted frame JPEGs, one subdirectory per source clip "
                        "(<output_dir>/<clip_stem>/<frame_idx>.jpg).")
    p.add_argument("--jpeg_quality", type=int, default=95,
                   help="PIL JPEG save quality. Video frames are already h264-lossy, so this doesn't "
                        "meaningfully add generation loss at typical qualities, and buys a large size "
                        "reduction versus PNG or keeping the full clip around.")
    p.add_argument("--force", action="store_true",
                   help="Bypass the skip-if-already-extracted resume check and re-extract every clip's "
                        "frames, overwriting whatever currently sits in --output_dir for it. Same idea "
                        "as download_panda70m_clips.py's --force_recut -- use for a one-time corrective "
                        "run (e.g. after re-cutting clips with the item-12 fix), not routine resubmits.")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Extract this many clips' frames in parallel via a process pool -- decord "
                        "decode + JPEG encode is CPU-bound (unlike download_panda70m_clips.py's "
                        "network-bound yt-dlp calls), so this uses processes, not threads, to actually "
                        "parallelize past the GIL. Match to --cpus-per-task.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process this many distinct clips (for a quick test run). Default: all.")
    return p.parse_args()


def load_targets(metadata_dir):
    """Groups TwiFF/Future-L1-50K rows by their `video` filename and returns
    {video_filename: sorted list of distinct frame indices} -- the union of
    every row referencing that clip's question_images_index and
    reasoning_images_index. Same required-column check
    TwiFFStage0Stream.__init__ already applies, so a schema mismatch fails
    loudly here rather than partway through extraction."""
    from datasets import load_from_disk

    rows = load_from_disk(metadata_dir)
    required_cols = {"video", "question_images_index", "reasoning_images_index"}
    missing = required_cols - set(rows.column_names)
    if missing:
        raise ValueError(
            f"metadata_dir={metadata_dir} is missing expected columns {missing} "
            f"(has {rows.column_names}) -- not a TwiFF/Future-L1-50K staged slice?"
        )

    by_video = {}
    n_bad_indices = 0
    for row in rows:
        video = row["video"]
        q_idx = row["question_images_index"] or []
        r_idx = row["reasoning_images_index"] or []
        if not q_idx and not r_idx:
            n_bad_indices += 1
            continue
        by_video.setdefault(video, set()).update(q_idx)
        by_video[video].update(r_idx)
    if n_bad_indices:
        log(f"  {n_bad_indices} row(s) had no question_images_index or reasoning_images_index -- skipped")
    return {video: sorted(idxs) for video, idxs in by_video.items()}


def already_extracted(out_dir, indices):
    """A clip counts as already-done if every requested frame index has a
    JPEG on disk under out_dir -- cheap file-existence checks, no decode."""
    return all(os.path.exists(os.path.join(out_dir, f"{i}.jpg")) for i in indices)


def extract_one(args_tuple):
    """Runs in a worker process. Opens video_path once via decord (same call
    pattern as train_stage0_latent_grounding.py's load_video_frames), pulls
    every requested frame index, and saves each as a JPEG. Returns a status
    dict -- never raises; any decode/file error is caught and reported so
    one bad clip can't kill the whole pool."""
    video_filename, indices, video_root, output_dir, jpeg_quality = args_tuple
    stem = video_filename[:-4] if video_filename.endswith(".mp4") else video_filename
    clip_out_dir = os.path.join(output_dir, stem)
    video_path = os.path.join(video_root, video_filename)

    if not os.path.exists(video_path):
        return {"video": video_filename, "status": "missing_file", "n_requested": len(indices), "n_extracted": 0}

    try:
        import decord
        from PIL import Image

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        n = len(vr)
        valid = [i for i in indices if 0 <= i < n]
        if not valid:
            return {"video": video_filename, "status": "out_of_range", "n_requested": len(indices),
                    "n_extracted": 0, "error": f"clip has {n} frames, wanted {indices}"}
        batch = vr.get_batch(valid).asnumpy()
        os.makedirs(clip_out_dir, exist_ok=True)
        for idx, frame in zip(valid, batch):
            Image.fromarray(frame).save(
                os.path.join(clip_out_dir, f"{idx}.jpg"), quality=jpeg_quality
            )
        skipped_indices = sorted(set(indices) - set(valid))
        status = "ok" if not skipped_indices else "partial"
        return {"video": video_filename, "status": status, "n_requested": len(indices),
                "n_extracted": len(valid), "skipped_indices": skipped_indices}
    except Exception as e:
        return {"video": video_filename, "status": "decode_failed", "n_requested": len(indices),
                "n_extracted": 0, "error": str(e)}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log("=" * 70)
    log(f"Loading targets from {args.metadata_dir}")
    log("=" * 70)
    by_video = load_targets(args.metadata_dir)
    log(f"{len(by_video)} distinct clip(s) referenced by this metadata slice")

    items = list(by_video.items())
    if args.limit is not None:
        items = items[:args.limit]

    pending = []
    n_already_done = 0
    for video_filename, indices in items:
        stem = video_filename[:-4] if video_filename.endswith(".mp4") else video_filename
        clip_out_dir = os.path.join(args.output_dir, stem)
        if not args.force and already_extracted(clip_out_dir, indices):
            n_already_done += 1
            continue
        pending.append((video_filename, indices, args.video_root, args.output_dir, args.jpeg_quality))

    log(f"{n_already_done} clip(s) already fully extracted, skipped. {len(pending)} clip(s) to process "
        f"with --num_workers={args.num_workers}"
        + (" (--force set: re-extracting even already-done clips)" if args.force else ""))

    stats = {"ok": 0, "partial": 0, "missing_file": 0, "out_of_range": 0, "decode_failed": 0}
    report_path = os.path.join(args.output_dir, "extract_report.jsonl")
    total_frames_written = 0

    with open(report_path, "a") as report_f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for i, result in enumerate(pool.map(extract_one, pending), start=1):
                stats[result["status"]] += 1
                total_frames_written += result.get("n_extracted", 0)
                report_f.write(json.dumps({**result, "time": time.time()}) + "\n")
                report_f.flush()
                if result["status"] in ("decode_failed", "missing_file", "out_of_range"):
                    log(f"  [{i}/{len(pending)}] {result['status']}: {result['video']} "
                        f"({result.get('error', '')})")
                elif i % 200 == 0:
                    log(f"  [{i}/{len(pending)}] ok so far: {stats['ok']} decode_failed: {stats['decode_failed']}")

    summary_path = os.path.join(args.output_dir, "extract_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "distinct_clips_in_metadata": len(by_video),
            "already_extracted_skipped": n_already_done,
            "processed_this_run": len(pending),
            "total_frames_written": total_frames_written,
            **stats,
        }, f, indent=2)

    log("=" * 70)
    log(f"DONE. already_done={n_already_done} " + " ".join(f"{k}={v}" for k, v in stats.items())
        + f" total_frames_written={total_frames_written}")
    log(f"Summary written to {summary_path}, per-clip report at {report_path}")
    log("=" * 70)
    if pending and stats["ok"] + stats["partial"] == 0:
        log("FATAL: zero clips extracted successfully this run -- inspect the failure counts above "
            "before trusting this output.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL: unhandled exception.")
        traceback.print_exc()
        sys.exit(1)
