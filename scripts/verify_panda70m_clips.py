#!/usr/bin/env python
"""
Spot-check the clips download_panda70m_clips.py fetched, before trusting
them for Stage 0 training.

Why this exists: download_panda70m_clips.py's whole "clipID is a 0-indexed
position into videoID's Panda-70M timestamp list" mapping is an inference
from one inspected example, not a documented guarantee (see that script's
module docstring, limitation 3). "The file downloaded without error" does
NOT confirm the clip is the right few seconds of the right video -- only
looking at it next to its TwiFF question/answer does. This script makes
that fast: for every clip actually on disk, it looks up the TwiFF row that
named it, checks the clip's real duration/resolution with ffprobe, and pulls
a few evenly-spaced frame thumbnails so you can eyeball content vs. the
question text without opening 12 separate video files.

What to actually do with the output
------------------------------------
1. Run this script (see Usage below).
2. Open spot_check_report.json (or just read the printed summary) and, for
   3-5 clips, look at the extracted thumbnails next to that row's
   `question`/`answer` text. Ask: does what's in the frame plausibly relate
   to the question being asked about this clip? Obvious mismatches (e.g. a
   question about a cooking action but the thumbnail shows a completely
   unrelated scene) are the signal that the clipID-as-index assumption is
   wrong for that row -- one or two misses in dark/ambiguous frames is
   normal, but a consistent pattern of unrelated content is not.
3. Also sanity-check clip duration: it should roughly match the [start, end]
   window Panda-70M's metadata gave that clip (a handful of seconds, not
   0.0s and not the full multi-minute source video -- either extreme means
   the ffmpeg cut step did something wrong even though it "succeeded").

This script does NOT re-download anything and needs no GPU -- it only reads
files download_panda70m_clips.py already wrote. Fine to run directly on the
login node (see Usage), no sbatch needed.

Usage
-----
    python verify_panda70m_clips.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/twiff_metadata_2000 \\
        --clips_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips \\
        --thumbnails_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips/_spot_check_thumbnails \\
        --num_thumbnails 3

Needs system `ffmpeg`/`ffprobe` (same env as download_panda70m_clips.sh
already set up) and the `datasets` package. No new pip installs needed if
you're running this in the same conda env.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata_dir", required=True,
                   help="Path to the staged TwiFF-2.7M slice (same one download_panda70m_clips.py used)")
    p.add_argument("--clips_dir", required=True,
                   help="Where download_panda70m_clips.py wrote clips (its --output_dir)")
    p.add_argument("--thumbnails_dir", default=None,
                   help="Where to write extracted frame thumbnails (default: <clips_dir>/_spot_check_thumbnails)")
    p.add_argument("--num_thumbnails", type=int, default=3,
                   help="Evenly-spaced frames to extract per clip for visual inspection")
    p.add_argument("--report_path", default=None,
                   help="Where to write the JSON report (default: <clips_dir>/spot_check_report.json)")
    return p.parse_args()


def ffprobe_info(path):
    """Returns (duration_seconds, width, height) for a video file, or (None, None, None) on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None, None, None
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0.0))
        streams = data.get("streams", [])
        width = streams[0]["width"] if streams else None
        height = streams[0]["height"] if streams else None
        return duration, width, height
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        return None, None, None


def extract_thumbnails(clip_path, duration, out_dir, stem, num_thumbnails):
    """Extracts num_thumbnails evenly-spaced frames as JPEGs. Returns list of written paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    if not duration or duration <= 0:
        return written
    # Evenly spaced timestamps, avoiding the very first/last frame (often black/transition frames).
    fractions = [(i + 1) / (num_thumbnails + 1) for i in range(num_thumbnails)]
    for i, frac in enumerate(fractions):
        ts = duration * frac
        out_path = os.path.join(out_dir, f"{stem}_frame{i}.jpg")
        cmd = ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", clip_path, "-frames:v", "1", "-q:v", "2", out_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(out_path):
            written.append(out_path)
    return written


def main():
    args = parse_args()
    thumbnails_dir = args.thumbnails_dir or os.path.join(args.clips_dir, "_spot_check_thumbnails")
    report_path = args.report_path or os.path.join(args.clips_dir, "spot_check_report.json")

    from datasets import load_from_disk
    log(f"Loading TwiFF metadata from {args.metadata_dir} ...")
    rows = load_from_disk(args.metadata_dir)
    by_video = {}
    for row in rows:
        by_video.setdefault(row["video"], row)  # first row wins if duplicated

    clip_files = sorted(
        f for f in os.listdir(args.clips_dir)
        if f.endswith(".mp4") and os.path.isfile(os.path.join(args.clips_dir, f))
    )
    log(f"Found {len(clip_files)} downloaded clip(s) in {args.clips_dir}")
    if not clip_files:
        log("Nothing to verify -- no .mp4 files in --clips_dir. Did the download step actually run?")
        sys.exit(1)

    report = []
    for fname in clip_files:
        clip_path = os.path.join(args.clips_dir, fname)
        stem = fname[:-4]
        twiff_row = by_video.get(fname)

        duration, width, height = ffprobe_info(clip_path)
        thumbs = extract_thumbnails(clip_path, duration, thumbnails_dir, stem, args.num_thumbnails)

        entry = {
            "video": fname,
            "duration_seconds": round(duration, 2) if duration else None,
            "resolution": f"{width}x{height}" if width and height else None,
            "thumbnails": thumbs,
            "found_in_twiff_metadata": twiff_row is not None,
            "question": twiff_row["question"] if twiff_row else None,
            "answer": twiff_row["answer"] if twiff_row else None,
        }
        report.append(entry)

        log("-" * 70)
        log(f"{fname}")
        log(f"  duration={entry['duration_seconds']}s  resolution={entry['resolution']}")
        if twiff_row is None:
            log("  WARNING: this filename wasn't found in the staged TwiFF metadata at all -- "
                "can't show a question/answer to compare against.")
        else:
            q = (twiff_row["question"] or "")[:200]
            a = (twiff_row["answer"] or "")[:200]
            log(f"  question: {q}")
            log(f"  answer:   {a}")
        if thumbs:
            log(f"  thumbnails: {', '.join(thumbs)}")
        else:
            log("  WARNING: no thumbnails extracted (duration missing/zero, or ffmpeg failed) -- "
                "check this clip by hand.")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log("=" * 70)
    log(f"DONE. Wrote {len(report)} entries to {report_path}")
    log(f"Thumbnails written under {thumbnails_dir}")
    log("Next: pull a few thumbnails + their printed question/answer down to your own machine "
        "(scp) and eyeball whether clip content plausibly matches the question. That's the real "
        "confirmation the clipID-as-index assumption held for this data, not just 'it downloaded'.")
    log("=" * 70)

    n_short_or_long = sum(
        1 for e in report if e["duration_seconds"] is not None and not (0.2 <= e["duration_seconds"] <= 60)
    )
    if n_short_or_long:
        log(f"NOTE: {n_short_or_long}/{len(report)} clip(s) have an unusual duration (<0.2s or >60s) -- "
            "worth checking those specifically, a Panda-70M clip should typically be a few seconds long.")


if __name__ == "__main__":
    main()
