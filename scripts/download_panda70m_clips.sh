#!/bin/bash
#SBATCH --job-name=download_panda70m_clips
#SBATCH --partition=long
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/download_panda70m_clips_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/download_panda70m_clips_%j.err
#SBATCH --time=47:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# Fetches the actual Panda-70M video clips TwiFF-2.7M's `video` column
# names, so train_stage0_latent_grounding.py has real video to train on.
# REWRITTEN 2026-09-07: cuts each clip using TwiFF's own meta_data.start/end
# directly (no more indexing into a separate Panda-70M metadata mirror --
# that approach was confirmed to produce WRONG clip content via visual
# spot-check; see download_panda70m_clips.py's module docstring for the
# full story and claude/latentvans-handoff.md's SLURM gotchas section for
# the root-cause writeup). No GPU needed -- this is a network/IO job.
#
# FULL-DATASET RUN as of 2026-09-07 (changed from a 150-clip pilot at user
# request): no --limit is passed below, so download_panda70m_clips.py
# fetches every distinct, usable clip in the staged metadata (see its own
# docstring's "Full-dataset default" section). Note this skips the
# originally-planned intermediate step of re-running a small pilot and
# visually spot-checking it before scaling up -- this meta_data-based
# rewrite has NEVER been verified against real downloaded content at any
# scale. Run verify_panda70m_clips.sh against the output as soon as this
# finishes (or even partway through, once a few dozen clips exist) and
# actually look at a sample of thumbnails before trusting this for
# training -- more so than for a pilot, precisely because a full run means
# far more bandwidth/time at stake if the rewrite still has a bug.
#
# --partition=long (cn1/cn4, 48h limit) instead of the old --partition=medium
# (24h) since this is no longer a small pilot. --time=47:00:00 (just under
# long's 48h ceiling) is a conservative budget, not a real estimate. If the
# job times out before finishing, JUST RESUBMIT THIS SAME SCRIPT (with the
# same SHARD_ID/NUM_SHARDS if you used sharding below):
# download_panda70m_clips.py skips any clip whose output file already
# exists, so a resubmit picks up where the last run left off rather than
# starting over or duplicating work. Check `sacct -j <id> --format=Elapsed`
# after this run to see how much of the 47h it actually needed.
#
# CORRECTIVE RE-CUT (2026-09-08): the 2026-09-07 full run above reused an
# --output_dir left over from an earlier, OLD/buggy pilot (pre meta_data
# rewrite), and its skip-if-exists resume logic silently preserved 144
# stale clips from that old pilot instead of re-cutting them correctly --
# confirmed via download_summary.json's already_exists:144 plus a direct
# content-based check (IJgJ4Nr292I_1.mp4's automated quality-screen
# MISMATCH cited "asparagus", absent from that row's TwiFF text, meaning
# the frames themselves were still stale). See download_panda70m_clips.py's
# module docstring, "CONFIRMED BUG (2026-09-08)", for the full writeup.
#
# To run the one-time corrective full re-cut (guarantees every target clip
# is freshly downloaded/cut under today's correct meta_data logic,
# overwriting anything already in --output_dir), submit with FORCE_RECUT=1:
#     FORCE_RECUT=1 sbatch download_panda70m_clips.sh
# This passes --force_recut through to the python script below. Do NOT set
# FORCE_RECUT=1 for routine resubmits after a timeout -- that defeats the
# whole point of the skip-if-exists resume logic and would re-download
# every clip from scratch for no reason. Leave it unset (the default) for
# normal runs/resumes.
#
# VALIDATION SET (added 2026-09-12): METADATA_DIR/OUTPUT_DIR below are now
# overridable via environment variables (defaulting to the existing train
# paths, so routine train-set resubmits are completely unaffected) so this
# same script can fetch clips for the official TwiFF-2.7M `validation`
# split too -- see data_staging.sh's own validation-split staging block and
# train_stage0_latent_grounding.py's --val_metadata_dir/--val_video_root
# flags, which this feeds into. To fetch validation clips:
#     METADATA_DIR=$SCRATCH/data/twiff_metadata_val \
#     OUTPUT_DIR=$SCRATCH/data/panda70m_clips_val \
#     sbatch download_panda70m_clips.sh
# This is a separate --output_dir on purpose -- validation clips must never
# land in the same directory as train clips, or a training run drawing on
# --video_root could accidentally include held-out videos, defeating the
# whole point of a held-out split.
#
# PARALLEL SHARDING (added 2026-09-16, because a single job was only
# clearing ~1013 clips in 6 hours against a 50,000-clip target -- far too
# slow in one job's wall-clock budget). If your account can run 2+ SLURM
# jobs concurrently, set NUM_SHARDS to the number of parallel jobs and
# SHARD_ID to which one each job is (0-indexed). All shards write into the
# SAME OUTPUT_DIR -- download_panda70m_clips.py partitions by distinct
# source video so shards never redownload the same video or race each
# other (see its own docstring, "PARALLEL SHARDING (2026-09-16)", for why
# this is not just "start from index N"). Submit both jobs against the
# SAME METADATA_DIR/OUTPUT_DIR, e.g. for 2 parallel shards on the
# Future-L1-50K train set:
#     METADATA_DIR=$SCRATCH/data/future_l1_50k_metadata_full \
#     OUTPUT_DIR=$SCRATCH/data/panda70m_clips \
#     NUM_SHARDS=2 SHARD_ID=0 \
#         sbatch download_panda70m_clips.sh
#     METADATA_DIR=$SCRATCH/data/future_l1_50k_metadata_full \
#     OUTPUT_DIR=$SCRATCH/data/panda70m_clips \
#     NUM_SHARDS=2 SHARD_ID=1 \
#         sbatch download_panda70m_clips.sh
# Each shard writes its own download_summary_shard{N}.json (not a shared
# download_summary.json -- two concurrent jobs writing the same file would
# race). To get the combined totals once both finish:
#     python -c '
#     import json, sys
#     totals = {}
#     for path in sys.argv[1:]:
#         with open(path) as f:
#             d = json.load(f)
#         for k in ("downloaded", "youtube_download_failed", "ffmpeg_cut_failed", "already_exists"):
#             totals[k] = totals.get(k, 0) + d[k]
#     print(totals)
#     ' "$OUTPUT_DIR"/download_summary_shard*.json
# If a sharded job times out or fails partway, resubmit that SAME
# SHARD_ID/NUM_SHARDS combination (not a different one) -- the partition is
# deterministic, so this resumes that shard's own work via the normal
# skip-if-exists logic instead of reshuffling which videos it owns.
# Leaving NUM_SHARDS unset (the default, 1) is the original single-job
# behavior and is completely unaffected by any of this.
#
# INTRA-JOB CONCURRENCY (added 2026-09-17): sharding alone (above) was
# still too slow -- a single unsharded job only cleared ~1013 clips in 6
# hours, so even 2 shards combined would take days for 50,000 clips. The
# real fix is CONCURRENCY: download_panda70m_clips.py now processes this
# many source videos at once (download + cut) within a single job/shard
# via a thread pool, instead of one at a time -- see its own docstring,
# "INTRA-JOB CONCURRENCY (2026-09-17)", for why this is safe (both yt-dlp
# and ffmpeg run as subprocesses, so threads genuinely parallelize here)
# and for the real trade-off (higher concurrency = higher request rate to
# YouTube = more risk of bot-detection/"Please sign in" failures, not a
# free win). Defaults to 4, matching --cpus-per-task=4 above -- if you
# raise CONCURRENCY, raise --cpus-per-task in the #SBATCH header to match,
# since ffmpeg's re-encode fallback is CPU-bound and oversubscribing will
# slow individual workers down rather than speed the job up. Combined with
# 2 shards, CONCURRENCY=4 is a real ~8x speedup over the original
# single-threaded, unsharded run -- watch the first hour of a run's log
# for a rise in "Please sign in" failures before pushing it higher.
#
# DECORD-BASED CUT VALIDATION, CORRECTED (2026-09-24): the decode-validation
# logic added 2026-09-09 checked a cut clip with `ffmpeg -v error -i ... -f
# null -`, on the theory that a cut which decodes cleanly under ffmpeg is a
# cut decord (the actual consumer, in both this pipeline's frame-extraction
# and training scripts) can read too. That theory was WRONG -- confirmed via
# a real clip decord skipped during job 61184's training run
# (mPYFaclQj6c-0:09:12.118-0:09:43.249.mp4), which passed that exact ffmpeg
# check cleanly (no stderr, exit 0) yet decord still could not open it.
# ffmpeg's own decoder tolerates a mid-GOP cut (missing reference frames
# from a stream-copy that landed between keyframes) that decord's stricter
# decoder rejects outright. download_panda70m_clips.py's cut_clip() now
# validates with decord itself (decord_can_read(), opens the cut result and
# reads sampled frames) instead of ffmpeg -- see the .py's own module
# docstring, item 5's "CORRECTED 2026-09-24" addendum, for the full
# writeup. decord is now a hard dependency of THIS script too (not just
# training/extraction) -- added to the pip install line below.
#
# TARGETED RECUT (added 2026-09-24, RECUT_FROM_LIST): re-cutting every one
# of ~42,214 clips with FORCE_RECUT=1 to pick up the corrected validation
# above would re-download/re-cut the ~60-70% of clips that were already
# fine, for no reason -- prohibitively expensive at this scale. If you
# already know which clips are currently broken (e.g. from
# extract_stage0_frames.sh's extract_report.jsonl -- filter to
# decode_failed/missing_file entries and write their `video` filenames, one
# per line, to a text file), pass that file's path as RECUT_FROM_LIST to
# re-download/re-cut ONLY those filenames, leaving every other already-good
# clip untouched:
#     RECUT_FROM_LIST=/scratch/users/anirban/tamaghnam/latentvans/data/broken_clips.txt \
#         sbatch download_panda70m_clips.sh
# Mutually exclusive with FORCE_RECUT -- set one or the other, not both.
# After a RECUT_FROM_LIST run finishes, re-run extract_stage0_frames.sh
# again to pick up the freshly-fixed clips (it already skips anything
# already-extracted, so it will naturally only reprocess the just-fixed
# subset).
#
# YOUTUBE "THE PAGE NEEDS TO BE RELOADED" / UNPLAYABLE FIX (added 2026-09-24):
# this is a DIFFERENT failure from the "Please sign in to confirm you're not
# a bot" case COOKIES_FILE targets below -- it means yt-dlp's JavaScript
# challenge solver (used to derive YouTube's per-request nsig/PO token)
# couldn't produce a valid result, so YouTube's player API returned an
# UNPLAYABLE status instead of real video data. As of the yt-dlp releases
# current at the time of this fix, yt-dlp no longer bundles its own JS
# interpreter for this -- it now delegates JS-challenge solving to an
# EXTERNAL JavaScript runtime via its "EJS" system (see yt-dlp's own wiki,
# "EJS"). Without one installed, YouTube extraction degrades to exactly this
# error on a real, non-trivial fraction of videos, independent of cookies.
# Two things are now required, both installed below, no root needed:
#   1. `yt-dlp[default]` instead of bare `yt-dlp` -- the `[default]` extra
#      bundles the EJS challenge-solving scripts with the pip package, so
#      this job never needs outbound network access to github/npm at
#      runtime just to fetch them (relevant on a cluster where egress may be
#      restricted job-to-job).
#   2. A real JS runtime on PATH for that bundled EJS code to actually run
#      in -- Deno is yt-dlp's default/recommended choice and needs no flags
#      to be picked up once it's on PATH. Installed via conda-forge (same
#      no-root pattern already used for ffmpeg below), not pip -- deno is a
#      standalone binary, not a Python package.
# This is unrelated to and does not replace COOKIES_FILE -- cookies address
# YouTube treating this account/IP as suspicious; EJS/Deno address yt-dlp's
# ability to solve YouTube's JS challenge at all. Both can be needed at
# once. If "The page needs to be reloaded" errors persist after this fix is
# in place, check the job log for a yt-dlp version/Deno-not-found warning
# near the top before assuming the fix didn't help -- and re-run
# `python -m pip install --upgrade "yt-dlp[default]"` manually, since
# YouTube's challenge format changes over time and this is a moving target,
# not a one-time fix.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"

METADATA_DIR="${METADATA_DIR:-$SCRATCH/data/twiff_metadata_2000}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/data/panda70m_clips}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
CONCURRENCY="${CONCURRENCY:-4}"
RECUT_FROM_LIST="${RECUT_FROM_LIST:-}"

echo "===== Download Panda-70M Clips (full dataset) ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "METADATA_DIR=$METADATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_SHARDS=$NUM_SHARDS SHARD_ID=$SHARD_ID"
echo "CONCURRENCY=$CONCURRENCY"

mkdir -p "$SCRATCH/logs" "$OUTPUT_DIR"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

# Deliberately does NOT touch torch/torchvision/torchaudio -- see
# claude/latentvans-handoff.md's SLURM gotchas section for exactly why an
# innocent-looking `pip install --upgrade` next to a pinned torch stack is
# dangerous in this env. yt-dlp/datasets have no torch dependency at all.
# decord added 2026-09-24: cut_clip()'s decord_can_read() validator needs
# it (see the long comment block above, "DECORD-BASED CUT VALIDATION").
# yt-dlp[default] (changed from bare yt-dlp, 2026-09-24): bundles the EJS
# JS-challenge scripts yt-dlp now needs for YouTube -- see "THE PAGE NEEDS
# TO BE RELOADED" comment block above.
python -m pip install --no-cache-dir --upgrade "yt-dlp[default]" datasets decord

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found on PATH -- installing via conda-forge..."
    conda install -y --override-channels -c conda-forge ffmpeg
fi

# Deno (added 2026-09-24): the JS runtime yt-dlp's bundled EJS scripts run
# in to solve YouTube's per-request JS challenge -- see "THE PAGE NEEDS TO
# BE RELOADED" comment block above. Installed via conda-forge, same no-root
# pattern as ffmpeg above; yt-dlp picks up `deno` on PATH automatically, no
# extra flags needed.
#
# --override-channels (added 2026-09-24, jobs 61346/61347): a plain
# `conda install -c conda-forge ...` still consults this account's
# configured default channels (repo.anaconda.com/pkgs/main, pkgs/r) as well
# as the explicit -c, and this cluster's conda now requires interactively
# accepting Anaconda's Terms of Service for those default channels before
# ANY conda install can proceed non-interactively -- which a batch SLURM
# job can never do, so both ffmpeg and deno installs above would hang/fail
# with CondaToSNonInteractiveError the first time either was actually
# missing. --override-channels restricts channel resolution to exactly the
# -c flags given (conda-forge only), so this never touches the ToS-gated
# default channels at all -- no interactive acceptance needed, nothing to
# ask the user to click through in a batch job.
if ! command -v deno >/dev/null 2>&1; then
    echo "deno not found on PATH -- installing via conda-forge (needed for yt-dlp's YouTube JS-challenge solving)..."
    conda install -y --override-channels -c conda-forge deno
fi

FORCE_RECUT_FLAG=()
if [[ "${FORCE_RECUT:-0}" == "1" ]]; then
    echo "FORCE_RECUT=1 -- passing --force_recut (one-time corrective re-cut: EVERY target clip will"
    echo "be re-downloaded/re-cut and will overwrite whatever is currently in \$OUTPUT_DIR)."
    FORCE_RECUT_FLAG=(--force_recut)
fi

RECUT_FROM_LIST_FLAG=()
if [[ -n "$RECUT_FROM_LIST" ]]; then
    if [[ "${FORCE_RECUT:-0}" == "1" ]]; then
        echo "FATAL: FORCE_RECUT=1 and RECUT_FROM_LIST are mutually exclusive -- set one or the other." >&2
        exit 1
    fi
    if [[ ! -f "$RECUT_FROM_LIST" ]]; then
        echo "FATAL: RECUT_FROM_LIST=$RECUT_FROM_LIST was set but that path does not exist." >&2
        exit 1
    fi
    echo "RECUT_FROM_LIST=$RECUT_FROM_LIST -- passing --recut_from_list (only the filenames listed"
    echo "in that file will be re-downloaded/re-cut; every other already-existing clip is left alone)."
    RECUT_FROM_LIST_FLAG=(--recut_from_list "$RECUT_FROM_LIST")
fi

# COOKIES_FILE (added 2026-09-14, at explicit user request to reduce YouTube
# bot-detection failures -- see download_panda70m_clips.py's own module
# docstring, "Known limitations" item 1): a browser-exported cookies.txt for
# a real, logged-in YouTube session, passed through to yt-dlp's --cookies.
# Reduces (does NOT eliminate) 403/bot-detection failures; has NO effect on
# private or removed/deleted videos -- those are permanently gone regardless.
# Blank (default) = unset, same behavior as before this change. To export
# one: use a browser extension like "Get cookies.txt LOCALLY" (or yt-dlp's
# own --cookies-from-browser on a machine where you can run yt-dlp directly)
# while logged into YouTube in that browser, save the file, and scp it to
# this cluster -- then point COOKIES_FILE at that path. Treat this file like
# a credential: it carries your YouTube session, so keep its permissions
# tight (chmod 600) and don't commit it anywhere.
COOKIES_FILE="${COOKIES_FILE:-}"
COOKIES_FLAG=()
if [[ -n "$COOKIES_FILE" ]]; then
    if [[ -f "$COOKIES_FILE" ]]; then
        echo "COOKIES_FILE=$COOKIES_FILE -- passing --cookies_file to reduce bot-detection failures."
        COOKIES_FLAG=(--cookies_file "$COOKIES_FILE")
    else
        echo "WARNING: COOKIES_FILE=$COOKIES_FILE was set but that path does not exist -- proceeding WITHOUT cookies." >&2
    fi
fi

python "$SCRATCH/scripts/download_panda70m_clips.py" \
    --metadata_dir "$METADATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$SHARD_ID" \
    --concurrency "$CONCURRENCY" \
    "${FORCE_RECUT_FLAG[@]}" \
    "${RECUT_FROM_LIST_FLAG[@]}" \
    "${COOKIES_FLAG[@]}"

if [[ "$NUM_SHARDS" == "1" ]]; then
    echo "===== Download Panda-70M clips (full dataset) completed — see $OUTPUT_DIR/download_summary.json ====="
else
    echo "===== Download Panda-70M clips shard $SHARD_ID/$NUM_SHARDS completed — see $OUTPUT_DIR/download_summary_shard${SHARD_ID}.json ====="
    echo "(Combine all shards' summaries once every shard has finished -- see this script's own comments, 'PARALLEL SHARDING'.)"
fi
echo "End: $(date)"
