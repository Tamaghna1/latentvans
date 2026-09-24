#!/bin/bash
#SBATCH --job-name=extract_stage0_frames
#SBATCH --partition=short
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/extract_stage0_frames_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/extract_stage0_frames_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -euo pipefail

# Extracts only the frames train_stage0_latent_grounding.py actually reads
# (question_images_index UNION reasoning_images_index, per clip) out of the
# already-downloaded Panda-70M clips in $VIDEO_ROOT, once, instead of
# leaving every training draw to live-decode the full clip via decord --
# see extract_stage0_frames.py's own module docstring for the full
# rationale (handoff.md item 12's ~30-41% sustained decord decode-failure
# rate, and the 2026-09-24 disk-quota-exceeded training crash). No GPU
# needed -- this is a local decode/CPU/disk job, no network calls (unlike
# download_panda70m_clips.sh).
#
# READ THIS BEFORE RUNNING: this does NOT reduce the underlying decord
# decode-failure rate, and does NOT by itself change
# train_stage0_latent_grounding.py's behavior -- see the .py's own module
# docstring, "WHAT THIS SCRIPT DOES NOT DO". If you want the failure rate
# itself fixed first, run the still-unapplied item-12 fix
# (FORCE_RECUT=1 sbatch download_panda70m_clips.sh, once its
# decode-validation fix is applied) BEFORE this, not after -- otherwise
# you extract from clips you're about to re-cut anyway and have to redo it.
#
# VALIDATION SET: same env-var override pattern as
# download_panda70m_clips.sh -- point METADATA_DIR/VIDEO_ROOT/OUTPUT_DIR at
# the val slice instead of train:
#     METADATA_DIR=$SCRATCH/data/twiff_metadata_val \
#     VIDEO_ROOT=$SCRATCH/data/panda70m_clips_val \
#     OUTPUT_DIR=$SCRATCH/data/panda70m_frames_val \
#         sbatch extract_stage0_frames.sh
#
# --time=12:00:00 is an unbenchmarked estimate, not a measured figure --
# this job does no network I/O at all (unlike download_panda70m_clips.sh),
# so it should be much faster than that script's multi-hour/day runs, but
# check `sacct -j <id> --format=Elapsed` after the first real run and
# adjust up or down.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"

METADATA_DIR="${METADATA_DIR:-$SCRATCH/data/future_l1_50k_metadata_full}"
VIDEO_ROOT="${VIDEO_ROOT:-$SCRATCH/data/panda70m_clips}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/data/panda70m_frames}"
NUM_WORKERS="${NUM_WORKERS:-8}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"

echo "===== Extract Stage 0 Frames ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "METADATA_DIR=$METADATA_DIR"
echo "VIDEO_ROOT=$VIDEO_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_WORKERS=$NUM_WORKERS"

mkdir -p "$SCRATCH/logs" "$OUTPUT_DIR"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

if [[ ! -d "$VIDEO_ROOT" ]]; then
    echo "WARNING: $VIDEO_ROOT does not exist yet -- every clip will be logged as missing_file." >&2
fi

FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
    echo "FORCE=1 -- passing --force (re-extracting every clip, overwriting existing frame JPEGs)."
    FORCE_FLAG=(--force)
fi

python "$SCRATCH/scripts/extract_stage0_frames.py" \
    --metadata_dir "$METADATA_DIR" \
    --video_root "$VIDEO_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers "$NUM_WORKERS" \
    --jpeg_quality "$JPEG_QUALITY" \
    "${FORCE_FLAG[@]}"

echo "===== Extract Stage 0 frames completed — see $OUTPUT_DIR/extract_summary.json ====="
echo "End: $(date)"
