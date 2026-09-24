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
   permissive than decord's threaded reader.
   FIX, written 2026-09-09 (validated against ffmpeg's own decode-through
   pass): required a clean `ffmpeg -v error -i dest_path -f null -` pass
   before accepting a stream-copy result, falling back to a full re-encode
   otherwise.
   CORRECTED 2026-09-24 -- the 2026-09-09 fix above was insufficient and
   this project has now confirmed it concretely, not just suspected it:
   spot-checking one clip that decord skipped during job 61184's real
   training run (mPYFaclQj6c-0:09:12.118-0:09:43.249.mp4) with the exact
   `ffmpeg -v error -i <file> -f null -` command decodes_cleanly() used
   came back completely clean -- no stderr, exit 0 -- even though decord
   cannot read this file. ffmpeg's own decoder is simply more permissive
   than decord's stricter/threaded one, so a clip can pass ffmpeg's
   decode-through check while still failing the ONE consumer this
   validation exists for. cut_clip() now validates with decord itself
   (decord_can_read(), below) instead of ffmpeg -- it actually opens the
   cut result with decord.VideoReader and reads a few sampled frames, the
   same way train_stage0_latent_grounding.py's load_video_frames() does,
   so "this function says a clip is good" and "decord can actually read
   it" are the same claim by construction. Both the stream-copy AND the
   re-encode fallback are now validated this way -- previously only the
   stream-copy path was checked at all, and the re-encode path was
   trusted unconditionally.
6. CONFIRMED BUG (2026-09-12, job 58430 against the validation split's
   1,447 targets): a `subprocess.run(..., timeout=...)` call that actually
   times out raises `subprocess.TimeoutExpired`, which was NOT caught
   anywhere -- it propagated all the way up to the top-level `except
   Exception` in `__main__` and killed the ENTIRE run at whatever video
   happened to be downloading/cutting when the timeout fired, rather than
   being treated like any other single-video failure (403, private,
   unavailable -- all of which WERE already handled gracefully via
   checking `result.returncode`). Confirmed for real: video 72/1447
   (`s0lCx-poYe0`) hit yt-dlp's 300s timeout at 20:20:44, exactly 5:00
   after that video's download started at 20:15:44, and the whole job
   died there -- losing no already-cut clips (skip-if-exists resume still
   works fine for `--force_recut`-less resubmits), but requiring a manual
   resubmit for every slow/hung download in a large batch instead of just
   logging one more failure and moving on. FIX: see download_source_video()
   and cut_clip()'s own docstrings below -- every subprocess.run() call
   with a timeout is now wrapped so a TimeoutExpired is treated exactly
   like any other failure (logged, counted, execution continues) instead
   of crashing the whole job.
7. CONFIRMED BUG (2026-09-16, Future-L1-50K download job): every single
   attempted download in this run failed with yt-dlp reporting
   "Incomplete YouTube ID <fragment>" for a truncated fragment of what
   should have been an 11-character video ID. Root cause: the video-ID
   parsing below used to be `stem.rsplit("_", 1)`, which assumes TwiFF's
   original `video` filename shape, `{videoID}_{clipID}.mp4` (e.g.
   `3jG5bbatGvk_7.mp4`). Eurayka/Future-L1-50K -- a different staged data
   source added to this project, see data_staging.sh's TRAIN_DATA_SOURCE
   toggle -- names its `video` field completely differently:
   `{videoID}-{start}-{end}.mp4` (e.g.
   `-0AAhXT99Zw-0:04:09.165-0:04:57.213.mp4`), using hyphens and colons
   instead of a single underscore separator. Splitting that shape on `_`
   or `-` is fundamentally unreliable regardless of which delimiter is
   picked, because real YouTube video IDs can themselves start with a
   hyphen or contain underscores and hyphens internally -- there is no
   delimiter that is guaranteed absent from the ID itself.
   FIX: YouTube video IDs are always exactly 11 characters, so
   `video_id` is now taken as a fixed 11-character prefix of `stem`
   instead of anything delimiter-based. This works identically for both
   TwiFF's original suffix shape and Future-L1-50K's -- in both cases the
   ID is the first 11 characters and everything after it is naming/
   timestamp information this function never needed for parsing anyway,
   since `start`/`end` were already always sourced from the row's own
   `meta_data`, never from the filename.

8. PARALLEL SHARDING (2026-09-16, added because a single job was only
   clearing ~1013 clips in 6 hours against a 50,000-clip target -- far too
   slow to finish in one job's wall-clock budget). The user's account can
   run 2 jobs concurrently, so this adds --shard_id/--num_shards instead of
   a "resume from index N" flag. A plain index is not a safe thing to hand
   a second, independently-started job here: targets are drawn from a
   `random.Random(seed).shuffle()` order that only exists inside a single
   process's run of load_twiff_targets(), so "index 1013" does not name a
   stable, reproducible position two separate invocations would agree on --
   and if both jobs simply started from the top of that same order with no
   partitioning at all, they would spend their first stretch fighting over
   the SAME early source videos (both yt-dlp-ing and ffmpeg-cutting the
   same handful of clips at the same time) before ever diverging, which is
   not real parallelism. Instead, --num_shards N / --shard_id K (0-indexed)
   deterministically partitions the distinct SOURCE VIDEOS (not raw target
   rows -- see below for why) so each shard downloads a disjoint subset
   from the very first video onward, with no overlap and no race between
   concurrently-running shards, while both still write into the same
   --output_dir (so the result is one combined clip set, no merge step
   needed afterward). Sharding by source video rather than by individual
   clip row keeps every clip that shares a source video in the same shard,
   so no shard ever redundantly re-downloads a raw video another shard
   already fetched. The partition is `index_in_by_video_dict % num_shards
   == shard_id`, computed AFTER grouping by source video and BEFORE any
   skip-if-exists filtering, so it is stable across resubmits of the same
   shard (a timed-out shard resubmitted with the same --shard_id/
   --num_shards picks up exactly where it left off, same as the existing
   single-job resume behavior). Each shard writes its own
   `download_summary_shard{shard_id}.json` instead of a shared
   `download_summary.json`, since two concurrent processes writing the same
   summary file would race and the last one to finish would silently
   overwrite the other's stats -- sum the per-shard files together (or see
   download_panda70m_clips.sh's own comments) to get the combined total.

9. INTRA-JOB CONCURRENCY (2026-09-17, added because even after sharding
   across 2 SLURM jobs the real bottleneck was still visible: a single
   unsharded job cleared only ~1013 clips in 6 hours (~169/hour), which
   over 2 shards is still only ~338/hour combined -- roughly 148 hours (6+
   days) for the full 50,000-clip Future-L1-50K train set. Sharding alone
   was not going to be enough. The actual reason each job is that slow: it
   downloads and cuts ONE source video at a time, fully sequentially, and
   the `yt-dlp` download of each video is network-bound (10-30+ seconds per
   video observed in real job logs, per-video, dominated by YouTube-side
   latency/throttling, not by this machine's CPU) -- while the `ffmpeg` cut
   that follows is comparatively fast. That means most of this job's
   wall-clock time is spent idle, waiting on one network call, when it
   could be waiting on several at once.
   --concurrency N (default 4, matching the sbatch template's
   --cpus-per-task=4) processes N source videos at a time within a single
   job/shard, via a thread pool -- safe here specifically because both
   yt-dlp and ffmpeg are invoked as *subprocesses* (subprocess.run), so
   each worker thread is blocked waiting on an OS-level child process, not
   holding Python's GIL, and genuinely runs in parallel with the others.
   Raw downloads already use a per-video_id tmp filename
   (_raw_downloads/{video_id}.mp4) and shards already guarantee disjoint
   video_id sets, so concurrent workers within one shard never touch the
   same file; a small lock only protects the shared stats counters and the
   "[n/total] cutting ..." progress line so those stay accurate instead of
   racing.
   TRADE-OFF, read before cranking --concurrency up: --sleep_between (see
   above) was written for a single-threaded run, where it throttles the
   WHOLE job to roughly 1 request every --sleep_between seconds. With
   --concurrency N, each worker still sleeps --sleep_between seconds after
   ITS OWN download, but N workers do this in parallel, so the job's real
   aggregate request rate becomes roughly N / --sleep_between requests/sec
   -- i.e. concurrency directly multiplies how fast this hits YouTube, not
   just how fast it hits disk. That is the whole point (more throughput),
   but it also means a too-high --concurrency is the most likely way to
   trade a faster job for a higher bot-detection/"Please sign in" failure
   rate (see "Known limitations" item 1). Start at the default (4) or
   --cpus-per-task's value, whichever is smaller -- do not set concurrency
   higher than --cpus-per-task without also raising --cpus-per-task in the
   sbatch header, since ffmpeg's re-encode fallback path (see cut_clip())
   is genuinely CPU-bound and oversubscribing that starves every worker
   rather than speeding any of them up. If bumping this ever visibly
   increases "Please sign in"/403 failures in a run's download_summary,
   that is this trade-off showing up for real -- dial --concurrency back
   down, not up.

10. TARGETED RECUT (2026-09-24, --recut_from_list). --force_recut (item 4
    above) is all-or-nothing: it bypasses skip-if-exists for EVERY target
    clip, which is the right tool for cleaning up a genuinely mixed-history
    --output_dir but a hugely expensive way to fix a known SUBSET of
    clips -- at Future-L1-50K's scale (42,214 clips), a full --force_recut
    means re-downloading and re-cutting the ~66-70% of clips that already
    decode fine, purely to reach the ~30-34% that don't (see item 5's
    2026-09-24 correction). --recut_from_list PATH takes a text file of
    exact `video` filenames (one per line, matching TwiFF's `video` column
    -- e.g. the `video` field of every `status: "decode_failed"` or
    `"missing_file"` entry in extract_stage0_frames.py's
    extract_report.jsonl) and re-downloads/re-cuts ONLY those, leaving
    every other already-existing clip untouched (counted as
    already_exists, same as the normal skip-if-exists path). Mutually
    exclusive with --force_recut. The intended workflow after applying
    item 5's decord_can_read() fix: run extract_stage0_frames.py once
    against the CURRENT (unfixed) clips to discover the failing set via
    its report, build a filenames-only list from the decode_failed/
    missing_file entries, pass that to --recut_from_list here, then
    re-run extract_stage0_frames.py again -- it already skips anything
    already extracted, so that second pass naturally only attempts the
    just-recut subset.

Usage
-----
    python download_panda70m_clips.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/twiff_metadata_2000 \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips

    # Two jobs in parallel against the SAME --output_dir (see "PARALLEL
    # SHARDING" above) -- submit both, e.g. via download_panda70m_clips.sh
    # with SHARD_ID=0/SHARD_ID=1 and NUM_SHARDS=2 (see that script's own
    # comments for the exact sbatch invocations):
    python download_panda70m_clips.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/future_l1_50k_metadata_full \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips \\
        --num_shards 2 --shard_id 0
    python download_panda70m_clips.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/future_l1_50k_metadata_full \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips \\
        --num_shards 2 --shard_id 1

    # One-time corrective re-cut of an --output_dir that may contain stale
    # clips from an older/buggy run (see "CONFIRMED BUG (2026-09-08)" above):
    python download_panda70m_clips.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/twiff_metadata_2000 \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips \\
        --force_recut

    # Targeted re-cut of just the clips decord can't read (see "TARGETED
    # RECUT (2026-09-24)" above) -- much cheaper than --force_recut at
    # Future-L1-50K's scale:
    python download_panda70m_clips.py \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/future_l1_50k_metadata_full \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips \\
        --recut_from_list /scratch/users/anirban/tamaghnam/latentvans/data/decord_failed_clips.txt

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

Needs system `ffmpeg` (checked up front), the `yt-dlp` + `datasets` Python
packages, and `decord` (checked up front, since cut_clip() now validates
every cut result by actually reading it back with decord -- see item 5's
2026-09-24 correction above) -- installed by download_panda70m_clips.sh,
not by this script (kept consistent with how every other template in this
project handles its own dependencies).
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
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
    p.add_argument("--download_timeout", type=int, default=300,
                   help="Seconds to allow one yt-dlp invocation before giving up on that source video "
                        "(see module docstring, 'CONFIRMED BUG (2026-09-12)'). A timeout here is now "
                        "treated as an ordinary per-video failure, logged and counted like a 403/"
                        "private/unavailable error, rather than crashing the whole run.")
    p.add_argument("--force_recut", action="store_true",
                   help="Bypass the skip-if-already-exists resume check and re-download + re-cut EVERY "
                        "target clip, overwriting whatever currently sits in --output_dir. Use this for "
                        "a one-time corrective run when --output_dir may contain stale clips left over "
                        "from an older/buggy version of this script -- see module docstring, 'CONFIRMED "
                        "BUG (2026-09-08)'. Do NOT use this for routine resume-after-timeout runs: that "
                        "is exactly what the default skip-if-exists behavior is for, and --force_recut "
                        "would re-download everything from scratch for no reason. Mutually exclusive "
                        "with --recut_from_list.")
    p.add_argument("--recut_from_list", default=None,
                   help="Path to a text file of exact `video` filenames (one per line) to re-download/"
                        "re-cut, leaving every other already-existing clip untouched -- see module "
                        "docstring, 'TARGETED RECUT (2026-09-24)'. Much cheaper than --force_recut when "
                        "only a known subset of clips (e.g. everything decord currently fails to read) "
                        "needs fixing. Mutually exclusive with --force_recut.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1,
                   help="Run this many independent shards of this job in parallel against the SAME "
                        "--output_dir, each downloading a disjoint subset of source videos -- see "
                        "module docstring, 'PARALLEL SHARDING (2026-09-16)'. Default 1 (no sharding, "
                        "unchanged single-job behavior).")
    p.add_argument("--shard_id", type=int, default=0,
                   help="Which shard this invocation is, 0-indexed (0 <= shard_id < num_shards). "
                        "Ignored when --num_shards is 1.")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Process this many distinct source videos at once (download + cut) within "
                        "this single job/shard, via a thread pool -- see module docstring, 'INTRA-JOB "
                        "CONCURRENCY (2026-09-17)'. Default 4, matching the sbatch template's "
                        "--cpus-per-task=4. Raising this multiplies how fast this job hits YouTube, "
                        "not just how fast it hits disk -- read that docstring section before setting "
                        "it above --cpus-per-task.")
    return p.parse_args()


def load_twiff_targets(metadata_dir, limit, seed):
    """Returns a list of (video_filename, videoID, start, end) tuples for up to `limit` TwiFF rows
    (or every usable row if `limit` is None -- the default, full-dataset behavior as of 2026-09-07),
    deterministically shuffled so a small explicit --limit still gives a representative pilot. `start`/
    `end` come straight from the row's own `meta_data` field -- see module docstring for why that's
    now the source of truth instead of a separate Panda-70M metadata lookup.

    VIDEO-ID PARSING FIX (2026-09-16): `video_id` is taken as a fixed 11-character prefix of the
    filename stem, not a delimiter split -- see module docstring, "CONFIRMED BUG (2026-09-16)". A
    YouTube video ID is always exactly 11 characters, and this holds for every `video` filename shape
    this project has staged so far (TwiFF original `{videoID}_{clipID}.mp4` and Future-L1-50K's
    `{videoID}-{start}-{end}.mp4`), since `start`/`end` for the actual clip cut always come from the
    row's own `meta_data`, never from parsing the filename suffix.
    """
    import random
    from datasets import load_from_disk

    rows = load_from_disk(metadata_dir)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    targets = []
    seen = set()
    n_bad_meta = 0
    n_bad_id = 0
    for i in order:
        row = rows[i]
        video_filename = row["video"]
        if video_filename in seen:
            continue  # TwiFF has multiple QA rows per clip in some cases; only need the file once
        seen.add(video_filename)
        stem = video_filename[:-4] if video_filename.endswith(".mp4") else video_filename
        if len(stem) < 11:
            n_bad_id += 1
            log(f"  WARNING: video filename stem shorter than a YouTube ID (11 chars): "
                f"{video_filename!r} -- skipping")
            continue
        video_id = stem[:11]  # YouTube video IDs are always exactly 11 characters -- see docstring above

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
    if n_bad_id:
        log(f"  {n_bad_id} row(s) skipped for a video filename too short to hold a YouTube ID")
    return targets


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def download_source_video(video_id, cookies_file, max_height, dest_path, timeout):
    """yt-dlp the source YouTube video for one videoID. Returns (ok, error_message).

    TIMEOUT FIX (2026-09-12): a `subprocess.run(..., timeout=...)` call that actually times out
    RAISES `subprocess.TimeoutExpired` instead of returning a nonzero exit code -- confirmed for real
    (job 58430) to have crashed the ENTIRE run at video 72/1447 because nothing caught it, hitting the
    top-level `except Exception` handler in `__main__` instead. Now caught explicitly and treated as
    an ordinary per-video failure (same shape as a 403/private/unavailable error, which the
    `result.returncode != 0` branch below already handled), so one slow/hung download costs this run
    one skipped video, not the whole job.
    """
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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"yt-dlp did not finish within {timeout}s -- timed out"
    if result.returncode != 0:
        return False, result.stderr.strip()[-500:]
    return True, ""


def decord_can_read(path):
    """Actually opens `path` with decord and reads a few sampled frames -- the SAME library and call
    pattern train_stage0_latent_grounding.py's load_video_frames() uses at training time. This is the
    real acceptance test for cut_clip() below, replacing the ffmpeg-based decodes_cleanly() check this
    function superseded.

    WHY (2026-09-24 correction -- see module docstring, item 5): the ffmpeg-based check
    (`ffmpeg -v error -i path -f null -`, clean exit + empty stderr required) was confirmed
    insufficient by direct evidence, not theory -- a real clip decord skipped during job 61184's
    training run (mPYFaclQj6c-0:09:12.118-0:09:43.249.mp4) passed that exact ffmpeg check cleanly.
    ffmpeg's own decoder is more permissive than decord's stricter/threaded one (this project's
    documented decord failure signatures -- "cannot find video stream", an EAGAIN from
    avcodec_send_packet -- are specifically about decord's stream probe/threaded decoder, not generic
    container corruption ffmpeg would also choke on). Validating with ffmpeg was answering "can ANY
    reasonably permissive tool decode this?" when the actual question is "can decord specifically read
    this?" -- those are different claims, and only this one is what actually matters for training.

    Reads first/middle/last frame (cheap -- a few decoded frames, not the whole clip) rather than
    every frame, since the goal is confirming decord CAN open and seek this file the way
    load_video_frames() will, not exhaustively validating every frame index up front (TwiFF/
    Future-L1-50K's question_images_index/reasoning_images_index are only ever a handful of specific
    indices anyway, checked at extraction/training time regardless).
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path)
        n = len(vr)
        if n == 0:
            return False
        probe_indices = sorted(set([0, n // 2, n - 1]))
        vr.get_batch(probe_indices).asnumpy()
        return True
    except Exception as e:
        log(f"    decord could not read the cut result ({e})")
        return False


def cut_clip(source_path, start, end, dest_path):
    """ffmpeg-cut [start, end] (HH:MM:SS.ms strings, straight from TwiFF's own meta_data) out of
    source_path. Tries a fast stream-copy cut first; falls back to re-encoding if that produces an
    empty/invalid file OR a file that only LOOKS fine by size but decord still can't read (see
    DECODE-VALIDATION FIX below). Always uses `-y`, so this overwrites dest_path if it already
    exists -- relied on by --force_recut/--recut_from_list.

    DECODE-VALIDATION FIX (2026-09-09, CORRECTED 2026-09-24): the old size-only check (
    "> 1024 bytes") wasn't enough -- a stream-copy cut (`-c copy`) landing on a non-keyframe boundary
    can produce a file well over 1024 bytes that plays back fine in permissive tools while still being
    broken enough that decord -- what train_stage0_latent_grounding.py actually uses to read frames --
    fails on it. The 2026-09-09 fix validated with ffmpeg's own `-v error -f null -` decode-through
    pass, which this project has since confirmed (2026-09-24, see module docstring item 5) is not a
    reliable predictor of whether decord specifically can read the result -- ffmpeg's decoder is more
    permissive. This now validates with decord_can_read() instead (actually opens the result with
    decord and reads sampled frames, the real consumer's own code path), and does so for BOTH the
    stream-copy attempt AND the re-encode fallback -- previously only the stream-copy path was
    validated at all, and the re-encode result was trusted unconditionally. If even the re-encode
    doesn't produce something decord can read, this clip is reported as a failed cut rather than
    silently accepted.

    TIMEOUT FIX (2026-09-12): both subprocess.run() calls below now catch `subprocess.TimeoutExpired`
    the same way download_source_video() does -- a hung ffmpeg process (e.g. a corrupt/huge source
    file) now just makes this clip count as a failed cut instead of crashing the whole run. See
    module docstring, "CONFIRMED BUG (2026-09-12)".
    """
    def run(extra_args):
        cmd = ["ffmpeg", "-y", "-ss", start, "-to", end, "-i", source_path] + extra_args + [dest_path]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            log(f"    ffmpeg cut timed out after 120s -- treating as a failed cut")
            return False
        return r.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024

    if run(["-c", "copy"]) and decord_can_read(dest_path):
        return True
    log(f"    stream-copy cut failed, produced a too-small file, or decord couldn't read it -- re-encoding instead")
    if run(["-c:v", "libx264", "-c:a", "aac"]) and decord_can_read(dest_path):
        return True
    log(f"    re-encoded result still isn't readable by decord -- reporting this clip as a failed cut")
    return False


def main():
    args = parse_args()

    if not ffmpeg_available():
        log("FATAL: `ffmpeg` not found on PATH. Install it in the conda env "
            "(conda install -y -c conda-forge ffmpeg) before running this script.")
        sys.exit(1)

    try:
        import decord  # noqa: F401 -- cut_clip()'s decord_can_read() needs this; check it up front
    except ImportError as e:
        log(f"FATAL: missing dependency ({e}). cut_clip() now validates every cut result with decord "
            f"(see module docstring, 'DECODE-VALIDATION FIX'). Try:\n"
            f"  python -m pip install --upgrade decord")
        sys.exit(1)

    if args.force_recut and args.recut_from_list:
        log("FATAL: --force_recut and --recut_from_list are mutually exclusive -- see module "
            "docstring, 'TARGETED RECUT (2026-09-24)'.")
        sys.exit(1)

    if args.num_shards < 1:
        log(f"FATAL: --num_shards must be >= 1, got {args.num_shards}")
        sys.exit(1)
    if not (0 <= args.shard_id < args.num_shards):
        log(f"FATAL: --shard_id must satisfy 0 <= shard_id < num_shards, got "
            f"shard_id={args.shard_id} num_shards={args.num_shards}")
        sys.exit(1)
    if args.concurrency < 1:
        log(f"FATAL: --concurrency must be >= 1, got {args.concurrency}")
        sys.exit(1)
    cpu_count = os.cpu_count()
    if cpu_count is not None and args.concurrency > cpu_count:
        log(f"WARNING: --concurrency={args.concurrency} exceeds os.cpu_count()={cpu_count} on this "
            f"node -- ffmpeg's re-encode fallback path is CPU-bound, so this can oversubscribe and "
            f"slow individual workers down rather than speeding the job up overall. See module "
            f"docstring, 'INTRA-JOB CONCURRENCY (2026-09-17)'.")

    recut_filenames = None
    if args.recut_from_list:
        with open(args.recut_from_list) as f:
            recut_filenames = {line.strip() for line in f if line.strip()}
        log(f"--recut_from_list: {len(recut_filenames)} filename(s) loaded from {args.recut_from_list} "
            f"-- only these will be (re-)downloaded/re-cut; every other existing clip is left untouched.")

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

    if recut_filenames is not None:
        all_filenames = {fn for fn, _vid, _s, _e in targets}
        n_not_found = len(recut_filenames - all_filenames)
        if n_not_found:
            log(f"WARNING: {n_not_found} of {len(recut_filenames)} --recut_from_list filename(s) were "
                f"not found among this run's targets (wrong --metadata_dir, or a typo/stale list?) -- "
                f"those entries will simply never match anything below.")

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

    # SHARDING (2026-09-16): partition by SOURCE VIDEO (not raw target row), and do it here --
    # after grouping, before skip-if-exists filtering -- so each shard's slice of work is stable
    # across resubmits and no two concurrently-running shards ever touch the same source video. See
    # module docstring, "PARALLEL SHARDING (2026-09-16)", for why a plain index into the shuffled
    # target order is not a safe thing to split two independent jobs on.
    all_video_items = list(by_video.items())
    if args.num_shards > 1:
        shard_video_items = [item for i, item in enumerate(all_video_items) if i % args.num_shards == args.shard_id]
        log(f"Sharding: this is shard {args.shard_id}/{args.num_shards} -- "
            f"{len(shard_video_items)}/{len(all_video_items)} distinct source videos assigned to it")
    else:
        shard_video_items = all_video_items

    stats = {"downloaded": 0, "youtube_download_failed": 0, "ffmpeg_cut_failed": 0, "already_exists": 0}
    stats_lock = threading.Lock()
    n_clips_done = 0  # protected by stats_lock -- shared progress counter across worker threads

    # Pre-filter to pending work sequentially (cheap: just os.path.exists checks, or a set lookup for
    # --recut_from_list) before handing video groups to the thread pool -- keeps "already_exists"
    # accounting simple and single-threaded, and means the pool only ever sees videos that actually
    # need a download.
    work_items = []  # (video_id, pending_clips) for videos with at least one clip still needed
    for video_id, clips in shard_video_items:
        if args.force_recut:
            # Corrective mode: treat every clip as pending regardless of what's already on disk --
            # see the --force_recut help text and module docstring for why this exists.
            pending = list(clips)
            already = 0
        elif recut_filenames is not None:
            # Targeted corrective mode (2026-09-24): only the filenames named in --recut_from_list are
            # pending, regardless of whether they currently exist on disk -- everything else in this
            # video's clip list is left exactly as-is (counted the same as an ordinary skip). See
            # module docstring, "TARGETED RECUT (2026-09-24)".
            pending = [(fn, s, e) for fn, s, e in clips if fn in recut_filenames]
            already = len(clips) - len(pending)
        else:
            pending = [(fn, s, e) for fn, s, e in clips if not os.path.exists(os.path.join(args.output_dir, fn))]
            already = len(clips) - len(pending)
        stats["already_exists"] += already
        n_clips_done += already
        if pending:
            work_items.append((video_id, pending))

    total_clips = sum(len(clips) for _video_id, clips in shard_video_items)
    log(f"{len(work_items)} source video(s) still need downloading (rest already on disk) -- "
        f"processing with --concurrency={args.concurrency}")

    def process_one_video(video_id, pending):
        """Runs in a worker thread: download this source video once, cut every pending clip from it,
        then delete the raw download. Returns a local stats dict -- callers sum these after the pool
        finishes rather than mutating shared `stats` directly from inside a worker, so only the small
        progress-counter/log section below needs a lock at all. See module docstring, 'INTRA-JOB
        CONCURRENCY (2026-09-17)'.
        """
        nonlocal n_clips_done
        local_stats = {"downloaded": 0, "youtube_download_failed": 0, "ffmpeg_cut_failed": 0}
        raw_path = os.path.join(tmp_dir, f"{video_id}.mp4")
        log(f"  yt-dlp {video_id} ({len(pending)} clip(s) needed) ...")
        ok, err = download_source_video(video_id, args.cookies_file, args.max_height, raw_path, args.download_timeout)
        time.sleep(args.sleep_between)
        if not ok:
            log(f"    FAILED: {err}")
            local_stats["youtube_download_failed"] += len(pending)
            with stats_lock:
                n_clips_done += len(pending)
            return local_stats

        for video_filename, start, end in pending:
            with stats_lock:
                n_clips_done += 1
                progress = n_clips_done
            dest_path = os.path.join(args.output_dir, video_filename)
            log(f"    [{progress}/{total_clips}] cutting {video_filename} ({start} -> {end})")
            if cut_clip(raw_path, start, end, dest_path):
                local_stats["downloaded"] += 1
            else:
                local_stats["ffmpeg_cut_failed"] += 1

        # Done with every clip that needed this source video -- free the scratch space now rather
        # than waiting for every other video in the run to finish too.
        try:
            os.remove(raw_path)
        except OSError:
            pass
        return local_stats

    log("=" * 70)
    log("Downloading + cutting clips (grouped by source video, "
        f"{args.concurrency} video(s) in flight at once)")
    log("=" * 70)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(process_one_video, video_id, pending) for video_id, pending in work_items]
        for future in concurrent.futures.as_completed(futures):
            # Any unexpected exception inside a worker (a bug, not an ordinary yt-dlp/ffmpeg failure --
            # those are already caught and returned as local_stats above) surfaces here via .result();
            # let it propagate to the top-level except Exception handler in __main__ rather than
            # silently swallowing it, same as the old single-threaded version would have.
            local_stats = future.result()
            for k, v in local_stats.items():
                stats[k] += v

    log("=" * 70)
    log("Cleaning up raw-download scratch dir (should already be empty -- per-video cleanup above)")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Separate summary file per shard (2026-09-16): two concurrently-running shards writing the
    # SAME download_summary.json would race, and whichever finishes last would silently overwrite
    # the other's stats -- see module docstring, "PARALLEL SHARDING (2026-09-16)". Unsharded runs
    # (the default, --num_shards 1) keep the original unsuffixed filename for backward compatibility.
    summary_filename = "download_summary.json" if args.num_shards == 1 else f"download_summary_shard{args.shard_id}.json"
    summary_path = os.path.join(args.output_dir, summary_filename)
    with open(summary_path, "w") as f:
        json.dump({"requested": total_clips, "distinct_source_videos": len(shard_video_items),
                   "shard_id": args.shard_id, "num_shards": args.num_shards,
                   "force_recut": args.force_recut, "recut_from_list": args.recut_from_list,
                   **stats}, f, indent=2)

    log("=" * 70)
    log(f"DONE. requested={total_clips} " + " ".join(f"{k}={v}" for k, v in stats.items()))
    log(f"Summary written to {summary_path}")
    log("=" * 70)
    if args.force_recut:
        log("This was a --force_recut run: already_exists should read 0 above, and every clip that "
            "succeeded was freshly re-downloaded/re-cut under today's correct meta_data logic and "
            "overwrote anything previously at that path. Re-run screen_panda70m_clip_quality.py "
            "against this output before trusting it for training.")
    elif args.recut_from_list:
        log(f"This was a --recut_from_list run: {stats['already_exists']} clip(s) not in the list "
            f"were left untouched; {stats['downloaded']} of the listed clip(s) were successfully "
            f"re-cut and are now decord-validated. Re-run extract_stage0_frames.py to pick up the "
            f"newly-fixed clips -- it already skips anything already extracted, so it will naturally "
            f"only attempt this just-recut subset.")
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
    if stats["downloaded"] == 0 and not (recut_filenames is not None and not work_items):
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
