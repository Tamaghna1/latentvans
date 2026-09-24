#!/usr/bin/env python
"""
Panda-70M clip downloader for LatentVANS -- fetches the actual video clips
that TwiFF-2.7M's `video` column names, so train_stage0_latent_grounding.py
has real video to train on instead of exiting with NoUsableExamplesError.

REWRITTEN 2026-09-07 after the original approach was confirmed WRONG.

What the original version did, and why it was wrong
------------------------------------------------------
The original script parsed TwiFF's `video` filename ("3jG5bbatGvk_7.mp4")
as `{videoID}_{clipID}.mp4` and used `clipID` as a 0-indexed position into
a *separately fetched* Panda-70M metadata mirror's (`multimodalart/panda-70m`)
per-video `timestamp` list. A visual spot-check (real thumbnails compared
against TwiFF's own `question`/`answer` text) found this produced the WRONG
clip content in 2 out of 2 checked "successful" downloads -- not a coverage
gap, not an edge case, just the wrong few seconds of video. Root cause:
Panda-70M's own GitHub documents that its smaller training splits (which is
what `multimodalart/panda-70m`'s `train_2m`/`train_10m` splits are built
from) keep only "at most 3 clips per source video" filtered by
`matching_score > 0.43` -- NOT the full original per-video clip list.
TwiFF's clipID values go well past 3 in this staged sample (e.g. `_7.mp4`),
which independently confirms TwiFF's clipID numbering was never meant to
index into that filtered, differently-ordered list. Indexing into it anyway
either falls out of range or lands on some other real (wrong) clip from the
same video.

The actual fix: TwiFF already tells you the clip window directly
---------------------------------------------------------------------
TwiFF-2.7M's own `meta_data` column (present in every row, previously
unused by any script in this project) is a dict that includes `start` and
`end` fields -- the exact "H:MM:SS.mmm" wall-clock timestamps for THIS
row's clip, plus a `Process`/`Summary` description that, when checked,
matches the row's `question`/`answer` text far better than anything the old
Panda-70M-mirror lookup produced. Confirmed 2026-09-07 by inspecting the
same two rows the visual spot-check had flagged as wrong:
    IJgJ4Nr292I_1.mp4 -> meta_data.Process: "...she is seen speaking to the
        camera... uses hand gestures to emphasize key points..." (matches
        the TwiFF answer) meta_data.start/end: 0:00:11.444 -> 0:00:24.657
        (~13.2s) -- nothing like the 3.23s clip the old approach cut.
    LDFhytZ7lX4_2.mp4 -> meta_data.Process: "...team in white attempting to
        score against team in blue... series of volleys and defensive
        moves..." meta_data.start/end: 0:00:37.504 -> 0:00:52.719 (~15.2s).
So this script now does NOT touch any external Panda-70M metadata source at
all -- it just cuts `[meta_data.start, meta_data.end]` out of the YouTube
video named by the videoID parsed from TwiFF's own `video` filename. Much
simpler, and grounded in TwiFF's own annotation instead of a
reverse-engineered index into someone else's differently-filtered mirror.

Known limitations of this approach (read before trusting a large run)
------------------------------------------------------------------------
1. YouTube-side attrition and bot-detection are real, currently-documented
   problems for this exact dataset (see snap-research/Panda-70M issue #60,
   "Sign in to confirm you're not a bot") -- expect a non-trivial failure
   rate independent of anything this script does wrong. Consider passing
   --cookies_file (a browser-exported cookies.txt) if failures look like
   bot-detection rather than "video removed".
2. `meta_data` is expected on every TwiFF row (confirmed present in the two
   rows checked, and required by train_stage0_latent_grounding.py's own
   column check), but this script still fails a row gracefully (counted
   separately) rather than crashing if `meta_data`/`start`/`end` is ever
   missing or malformed on some row -- schemas can have edge cases beyond
   what two inspected rows can prove.
3. WORTH RE-CONFIRMING (this rewrite hasn't been run yet): re-run
   verify_panda70m_clips.py against a fresh pilot from this rewritten
   script and spot-check a few clips again. The two rows checked above look
   right on paper (meta_data text matches the question/answer, and duration
   is sane), but "looks right in the metadata" and "the cut mp4 is actually
   right" are different claims until re-verified visually.
4. CONFIRMED BUG (2026-09-08): the "skip if destination file already
   exists" resume logic below (see `pending = [...]` in main()) is safe
   for its original purpose -- resuming a run that timed out partway,
   where existing files are already-correct clips cut by this same
   corrected script -- but it is NOT safe if `--output_dir` is reused
   across a different, older/buggy run of this script. This project hit
   exactly that: the full 2026-09-07 run reused the same --output_dir as
   an earlier pilot that used the OLD (confirmed-wrong) Panda-70M-mirror-
   index approach described above. `download_summary.json` from that run
   recorded `already_exists: 144` -- meaning 144 clips were silently
   skipped and left as whatever the OLD, wrong logic had cut, never
   re-cut under this file's correct meta_data.start/end logic. Confirmed
   concretely, not just suspected: the automated quality screen
   (screen_panda70m_clip_quality.py) flagged IJgJ4Nr292I_1.mp4 -- one of
   the two rows named above -- as a MISMATCH citing "asparagus", a word
   appearing nowhere in that row's TwiFF question/answer text. The VLM
   could only have gotten that from actually seeing asparagus in frames
   still left over from the original wrong-clip pilot.
   FIX: pass --force_recut (see below) for a one-time corrective run.
   That bypasses the skip-if-exists check entirely, so every target clip
   is freshly re-downloaded and re-cut and overwrites whatever is
   currently in --output_dir (cut_clip()'s ffmpeg call already uses `-y`,
   so no change was needed there). Do NOT pass --force_recut for routine
   resume-after-timeout runs -- that defeats the whole point of the
   skip-if-exists logic and re-downloads everything from scratch for no
   reason. It exists specifically to clean up an --output_dir with a
   mixed history like this one, once.
5. CONFIRMED BUG (2026-09-09): cut_clip()'s fast stream-copy path
   (`-c copy`) only checked that the output file existed and was over
   1024 bytes -- not that it actually decodes cleanly. A stream-copy cut
   landing on a non-keyframe boundary can produce a file well over 1024
   bytes that plays back fine in permissive tools (ffplay, the ffmpeg-
   based thumbnails verify_panda70m_clips.py extracts, and whatever
   frame-sampling screen_panda70m_clip_quality.py's judge used) while
   still being broken enough that decord -- what
   train_stage0_latent_grounding.py actually uses to read frames --
   fails on it: either "cannot find video stream" (decord's stream probe
   rejects the container) or an EAGAIN from its threaded decoder
   mid-decode. Confirmed for real: job 57910, the first real Stage 0
   training run against this output, hit both error signatures on
   roughly 38% of clip draws by step 20 -- none of which had been caught
   by any earlier verification step, because all of those were more
   permissive than decord's threaded reader. FIX: see cut_clip()'s own
   docstring below (a full decode-through validation pass before
   accepting a stream-copy result). NOT yet confirmed against a real
   batch -- no cluster access from where this was written. If you
   --force_recut with this fix, re-run screen_panda70m_clip_quality.py
   and then check the decord skip rate on a resubmitted training job
   actually drops before trusting this fix.

Usage
-----
    python download_panda70m_clips.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/twiff_metadata_2000 \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips

    # One-time corrective re-cut of an --output_dir that may contain stale
    # clips from an older/buggy run (see "CONFIRMED BUG (2026-09-08)" above):
    python download_panda70m_clips.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/twiff_metadata_2000 \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips \\
        --force_recut

Full-dataset default (changed 2026-09-07, at user request)
------------------------------------------------------------
`--limit` now defaults to None -- every distinct, usable clip in the staged
TwiFF sample (up to ~2000, minus any rows missing meta_data.start/end),
not a small pilot subset. Pass an explicit `--limit N` to go back to a small
pilot run (e.g. while testing a further code change). Two consequences of
running at this scale, both handled in this rewrite:
  - Raw per-video YouTube downloads are now deleted as soon as all of that
    video's clips are cut, instead of only once at the very end of the
    whole run -- otherwise thousands of raw source videos could pile up in
    scratch simultaneously. Clips are also grouped by source video so a
    video needed for multiple clips is only fetched from YouTube once.
  - This will take much longer than the old 150-clip pilot (see
    download_panda70m_clips.sh's own comments for a time-budget estimate
    and why the run is safe to simply resubmit if it times out partway --
    already-downloaded clips are skipped, not re-fetched, unless
    --force_recut is passed).
This full run has NOT been preceded by a successful, re-verified pilot of
this meta_data-based rewrite -- see "Known limitations" item 3 below. That
was the original plan (re-run a 150-clip pilot, verify with
verify_panda70m_clips.py, then scale up); running the full dataset directly
skips that intermediate check, so verifying a real sample of the output
before trusting it for training matters more here, not less.

Needs system `ffmpeg` (checked up front) and the `yt-dlp` + `datasets`
Python packages (installed by download_panda70m_clips.sh, not by this
script -- kept consistent with how every other template in this project
handles its own dependencies).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata_dir", required=True,
                   help="Path to the staged TwiFF-2.7M slice (the save_to_disk dir data_staging.sh produced)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write clips, named exactly as TwiFF's `video` column expects "
                        "(this becomes train_stage0_latent_grounding.py's --video_root)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only attempt this many TwiFF rows. Default (unset) is 'no limit' -- every "
                        "distinct clip in --metadata_dir's staged rows that has usable meta_data.start/"
                        "end. This became the default 2026-09-07 at the user's request to run the full "
                        "staged dataset rather than a small pilot subset; pass an explicit --limit if "
                        "you want a small pilot again (e.g. for testing a code change).")
    p.add_argument("--cookies_file", default=None,
                   help="Path to a browser-exported cookies.txt for yt-dlp, to reduce YouTube "
                        "bot-detection failures (see module docstring, limitation 1). Optional.")
    p.add_argument("--sleep_between", type=float, default=1.5,
                   help="Seconds to sleep between yt-dlp invocations for distinct source videos, as "
                        "a mild rate limit against triggering bot-detection")
    p.add_argument("--max_height", type=int, default=480,
                   help="Cap on downloaded video height -- we only need a few-second clip that gets "
                        "downsampled again by the VLM's own image processor, so there is no reason to "
                        "pull full 1080p/4K source video")
    p.add_argument("--force_recut", action="store_true",
                   help="Bypass the skip-if-already-exists resume check and re-download + re-cut EVERY "
                        "target clip, overwriting whatever currently sits in --output_dir. Use this for "
                        "a one-time corrective run when --output_dir may contain stale clips left over "
                        "from an older/buggy version of this script -- see module docstring, 'CONFIRMED "
                        "BUG (2026-09-08)'. Do NOT use this for routine resume-after-timeout runs: that "
                        "is exactly what the default skip-if-exists behavior is for, and --force_recut "
                        "would re-download everything from scratch for no reason.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_twiff_targets(metadata_dir, limit, seed):
    """Returns a list of (video_filename, videoID, start, end) tuples for up to `limit` TwiFF rows
    (or every usable row if `limit` is None -- the default, full-dataset behavior as of 2026-09-07),
    deterministically shuffled so a small explicit --limit still gives a representative pilot. `start`/
    `end` come straight from the row's own `meta_data` field -- see module docstring for why that's
    now the source of truth instead of a separate Panda-70M metadata lookup."""
    import random
    from datasets import load_from_disk

    rows = load_from_disk(metadata_dir)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    targets = []
    seen = set()
    n_bad_meta = 0
    for i in order:
        row = rows[i]
        video_filename = row["video"]
        if video_filename in seen:
            continue  # TwiFF has multiple QA rows per clip in some cases; only need the file once
        seen.add(video_filename)
        stem = video_filename[:-4] if video_filename.endswith(".mp4") else video_filename
        if "_" not in stem:
            log(f"  WARNING: unexpected video filename shape (no '_'): {video_filename!r} -- skipping")
            continue
        video_id, _clip_id_str = stem.rsplit("_", 1)  # clip_id no longer used for lookup, only naming

        meta = row.get("meta_data") if hasattr(row, "get") else row["meta_data"]
        start = meta.get("start") if meta else None
        end = meta.get("end") if meta else None
        if not start or not end:
            n_bad_meta += 1
            log(f"  WARNING: {video_filename!r} has no usable meta_data.start/end -- skipping")
            continue

        targets.append((video_filename, video_id, start, end))
        if limit is not None and len(targets) >= limit:
            break
    if n_bad_meta:
        log(f"  {n_bad_meta} row(s) skipped for missing/malformed meta_data.start/end")
    return targets


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def download_source_video(video_id, cookies_file, max_height, dest_path):
    """yt-dlp the source YouTube video for one videoID. Returns True on success."""
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
        "-o", dest_path,
        "--no-progress",
        "--quiet",
        "--no-warnings",
    ]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return False, result.stderr.strip()[-500:]
    return True, ""


def cut_clip(source_path, start, end, dest_path):
    """ffmpeg-cut [start, end] (HH:MM:SS.ms strings, straight from TwiFF's own meta_data) out of
    source_path. Tries a fast stream-copy cut first; falls back to re-encoding if that produces an
    empty/invalid file OR a file that only LOOKS fine by size but doesn't actually decode cleanly
    (see DECODE-VALIDATION FIX below). Always uses `-y`, so this overwrites dest_path if it already
    exists -- relied on by --force_recut.

    DECODE-VALIDATION FIX (2026-09-09): the old size-only check ("> 1024 bytes") wasn't enough. A
    stream-copy cut (`-c copy`) landing on a non-keyframe boundary can produce a file well over 1024
    bytes that plays back fine in permissive tools (ffplay, the ffmpeg-based thumbnails
    verify_panda70m_clips.py extracts, whatever frame sampling screen_panda70m_clip_quality.py's
    judge used) while still being broken enough that decord -- what
    train_stage0_latent_grounding.py actually uses to read frames -- fails on it, either with
    "cannot find video stream" (decord's stream probe rejects the container) or an EAGAIN from its
    threaded decoder mid-decode. Confirmed for real: job 57910's first real Stage 0 training run hit
    both error signatures on ~38% of clip draws by step 20, none of which any earlier verification
    step had caught (all more permissive than decord's threaded reader). Now, after a stream-copy
    attempt looks OK by file size, this also runs `ffmpeg -v error -i dest_path -f null -` (a full
    silent decode-through pass, discarding output) and requires a clean exit with no stderr output
    before accepting the stream-copy result; anything else falls through to the re-encode path,
    which normalizes keyframe placement and sidesteps this class of problem entirely. This has NOT
    been tested against a real batch of Panda-70M clips (no cluster access from where this was
    written) -- if you --force_recut with this and the decord skip rate on a resubmitted training
    job doesn't drop, the real failure mode may be something this check doesn't catch.
    """
    def run(extra_args):
        cmd = ["ffmpeg", "-y", "-ss", start, "-to", end, "-i", source_path] + extra_args + [dest_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024

    def decodes_cleanly(path):
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0 and not r.stderr.strip()

    if run(["-c", "copy"]) and decodes_cleanly(dest_path):
        return True
    log(f"    stream-copy cut failed, produced a too-small file, or didn't decode cleanly -- re-encoding instead")
    return run(["-c:v", "libx264", "-c:a", "aac"])


def main():
    args = parse_args()

    if not ffmpeg_available():
        log("FATAL: `ffmpeg` not found on PATH. Install it in the conda env "
            "(conda install -y -c conda-forge ffmpeg) before running this script.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_raw_downloads")
    os.makedirs(tmp_dir, exist_ok=True)

    log("=" * 70)
    log(f"Loading {'up to ' + str(args.limit) if args.limit is not None else 'all'} "
        f"TwiFF targets from {args.metadata_dir}")
    log("=" * 70)
    targets = load_twiff_targets(args.metadata_dir, args.limit, args.seed)
    log(f"{len(targets)} distinct clip files targeted for this run "
        f"(start/end taken directly from each row's own meta_data)")

    if args.force_recut:
        log("--force_recut is set: the skip-if-already-exists resume check is DISABLED. Every "
            "target clip will be freshly re-downloaded/re-cut and will OVERWRITE whatever is "
            "currently in --output_dir. This is meant for a one-time corrective run over an "
            "--output_dir that may hold stale clips from an older/buggy run -- see module "
            "docstring, 'CONFIRMED BUG (2026-09-08)'.")

    # Group by source video so every clip cut from a given YouTube video is done back-to-back, and
    # that video's raw download can be deleted as soon as its clips are done. At --limit 150 this
    # didn't matter (the whole tmp_dir was small enough to just delete once at the end), but at full-
    # dataset scale (thousands of distinct source videos, each up to --max_height quality) keeping
    # every raw download around until the very end could accumulate a large amount of scratch space
    # for no reason -- bounding it to "at most a handful of raw videos in flight at once" instead.
    by_video = {}
    for video_filename, video_id, start, end in targets:
        by_video.setdefault(video_id, []).append((video_filename, start, end))
    log(f"{len(by_video)} distinct source YouTube videos among those targets")

    stats = {"downloaded": 0, "youtube_download_failed": 0, "ffmpeg_cut_failed": 0, "already_exists": 0}

    log("=" * 70)
    log("Downloading + cutting clips (grouped by source video)")
    log("=" * 70)
    n_clips_done = 0
    total_clips = len(targets)
    for v_idx, (video_id, clips) in enumerate(by_video.items()):
        if args.force_recut:
            # Corrective mode: treat every clip as pending regardless of what's already on disk --
            # see the --force_recut help text and module docstring for why this exists.
            pending = list(clips)
            already = 0
        else:
            pending = [(fn, s, e) for fn, s, e in clips if not os.path.exists(os.path.join(args.output_dir, fn))]
            already = len(clips) - len(pending)
        stats["already_exists"] += already
        n_clips_done += already
        if not pending:
            continue  # every clip from this source video already exists on disk

        raw_path = os.path.join(tmp_dir, f"{video_id}.mp4")
        log(f"  [video {v_idx+1}/{len(by_video)}] yt-dlp {video_id} ({len(pending)} clip(s) needed) ...")
        ok, err = download_source_video(video_id, args.cookies_file, args.max_height, raw_path)
        time.sleep(args.sleep_between)
        if not ok:
            log(f"    FAILED: {err}")
            stats["youtube_download_failed"] += len(pending)
            n_clips_done += len(pending)
            continue

        for video_filename, start, end in pending:
            n_clips_done += 1
            dest_path = os.path.join(args.output_dir, video_filename)
            log(f"    [{n_clips_done}/{total_clips}] cutting {video_filename} ({start} -> {end})")
            if cut_clip(raw_path, start, end, dest_path):
                stats["downloaded"] += 1
            else:
                stats["ffmpeg_cut_failed"] += 1

        # Done with every clip that needed this source video -- free the scratch space now rather
        # than waiting for every other video in the run to finish too.
        try:
            os.remove(raw_path)
        except OSError:
            pass

    log("=" * 70)
    log("Cleaning up raw-download scratch dir (should already be empty -- per-video cleanup above)")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    summary_path = os.path.join(args.output_dir, "download_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"requested": len(targets), "distinct_source_videos": len(by_video),
                   "force_recut": args.force_recut, **stats}, f, indent=2)

    log("=" * 70)
    log(f"DONE. requested={len(targets)} " + " ".join(f"{k}={v}" for k, v in stats.items()))
    log(f"Summary written to {summary_path}")
    log("=" * 70)
    if args.force_recut:
        log("This was a --force_recut run: already_exists should read 0 above, and every clip that "
            "succeeded was freshly re-downloaded/re-cut under today's correct meta_data logic and "
            "overwrote anything previously at that path. Re-run screen_panda70m_clip_quality.py "
            "against this output before trusting it for training.")
    else:
        log("IMPORTANT: run verify_panda70m_clips.py against this output before trusting it -- this "
            "script was just rewritten to use TwiFF's own meta_data.start/end instead of the old, "
            "confirmed-wrong Panda-70M-mirror index lookup, and that rewrite has never been verified "
            "against real downloaded content at any scale before this run. Spot-check a real sample of "
            "thumbnails (not just a couple) before trusting this for training, given the full-dataset "
            "size of this run. See this script's module docstring. If `already_exists` above is "
            "nonzero and --output_dir was ever used by an older version of this script, re-run with "
            "--force_recut to guarantee no stale clips survive -- see module docstring, 'CONFIRMED "
            "BUG (2026-09-08)'.")
    if stats["downloaded"] == 0:
        log("FATAL: zero clips downloaded successfully -- inspect the failure counts above before "
            "trusting or reusing this output.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL: unhandled exception.")
        traceback.print_exc()
        sys.exit(1)
