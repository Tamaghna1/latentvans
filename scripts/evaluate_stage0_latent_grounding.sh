#!/bin/bash
#SBATCH --job-name=evaluate_stage0_latent_grounding
#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/evaluate_stage0_latent_grounding_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/evaluate_stage0_latent_grounding_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
set -euo pipefail

# Runs evaluate_stage0_latent_grounding.py -- see that file's module
# docstring for the full story. Short version: this does NOT train anything.
# It re-evaluates every step_<N> checkpoint already saved by a completed
# train_stage0_latent_grounding.sh run against the TwiFF-2.7M official
# validation split, and writes eval_metrics.jsonl plus a
# val_vs_train_loss.png comparison plot -- the "does val loss track train
# loss down, or plateau/diverge" graph flagged as the whole point of adding
# held-out validation (see claude/latentvans-handoff.md items 13/14).
#
# PREREQUISITE, same as train_stage0_latent_grounding.sh's own validation
# support: the validation split's clips must already be staged. If you
# haven't done this yet:
#   1. sbatch data_staging.sh                          (stages twiff_metadata_val, if not already)
#   2. METADATA_DIR=$SCRATCH/data/twiff_metadata_val \
#      OUTPUT_DIR=$SCRATCH/data/panda70m_clips_val \
#      sbatch download_panda70m_clips.sh
#   3. METADATA_DIR=$SCRATCH/data/twiff_metadata_val \
#      CLIPS_DIR=$SCRATCH/data/panda70m_clips_val \
#      sbatch screen_panda70m_clip_quality.sh
# Only after (3) will this script find usable .mp4 files under
# $VAL_VIDEO_ROOT -- otherwise every checkpoint will log n_eval=0 and
# nothing will get plotted.
#
# --mem=90G / --partition=ada: same reasoning as train_stage0_latent_grounding.sh
# (see claude/latentvans-handoff.md's ada partition notes) -- this job loads
# the same 3B model, just without an optimizer/gradient state, so it should
# fit comfortably, but hasn't been separately measured -- check
# `sacct -j <id> --format=MaxRSS` after the first run and adjust.
#
# --time=06:00:00: this reloads the FULL base model fresh for EACH checkpoint
# evaluated (see the .py's own docstring for why -- correctness over
# cleverness for a one-off eval script) plus up to EVAL_MAX_EXAMPLES forward
# passes per checkpoint. For 10 checkpoints (a default 2000-step run with
# --save_every 200) x 200 eval examples, this is a rough, unmeasured budget,
# not a real estimate -- there's no prior run of this script to time it
# against. Safe to just resubmit with --steps set to only the checkpoints
# not yet in eval_metrics.jsonl if it times out partway (this script APPENDS
# to eval_metrics.jsonl, it doesn't overwrite).
#
# CHECKPOINT_ROOT below defaults to stage0_run1 -- the --output_dir name
# used in train_stage0_latent_grounding.py's own usage example. Change it if
# your actual training run used a different --output_dir.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"

QWEN_DIR="$SCRATCH/checkpoints/Qwen2.5-VL-3B-Instruct"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRATCH/checkpoints/stage0_run1}"
VAL_METADATA_DIR="${VAL_METADATA_DIR:-$SCRATCH/data/twiff_metadata_val}"
VAL_VIDEO_ROOT="${VAL_VIDEO_ROOT:-$SCRATCH/data/panda70m_clips_val}"
VAL_EXCLUDE_FLAGGED_JSON="${VAL_EXCLUDE_FLAGGED_JSON:-$VAL_VIDEO_ROOT/quality_screen_flagged.json}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-200}"
STEPS="${STEPS:-}"                     # e.g. STEPS=200,600,1000,1400,1800,2000 to subsample
WANDB_PROJECT="stage0"      # blank = off, same opt-in pattern as training
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
# WANDB_RESUME_RUN_ID (added 2026-09-13): set this to the ORIGINAL training run's W&B run ID
# (from its URL, wandb.ai/<entity>/<project>/runs/<this-id> -- e.g. the stage0-57910 run) to have
# val_loss_* land on that SAME run's chart instead of a separate fresh eval run -- the actual
# train-vs-val overlay in one dashboard. Leave blank for a fresh eval run (the safe default).
WANDB_RESUME_RUN_ID="${WANDB_RESUME_RUN_ID:-}"

echo "===== Stage 0 -- Validation-Only Evaluation ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "VAL_METADATA_DIR=$VAL_METADATA_DIR"
echo "VAL_VIDEO_ROOT=$VAL_VIDEO_ROOT"
echo "EVAL_MAX_EXAMPLES=$EVAL_MAX_EXAMPLES"
nvidia-smi || true

mkdir -p "$SCRATCH/logs"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

# Same defensive LD_LIBRARY_PATH prepend as every other GPU job template in
# this project -- see vlm_vdm_bridge_smoke_test.sh / claude/latentvans-handoff.md
# for the full ncclCommResume story. Does NOT touch torch/torchvision/torchaudio.
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

# matplotlib has no torch dependency -- safe to plain-upgrade, unlike
# torchvision (see claude/latentvans-handoff.md's dependency-pinning-gap
# lesson for exactly why that distinction matters in this project).
python -m pip install --no-cache-dir --upgrade matplotlib

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT does not exist -- nothing to evaluate." >&2
    exit 1
fi

if [[ ! -d "$VAL_VIDEO_ROOT" ]] || [[ -z "$(ls -A "$VAL_VIDEO_ROOT"/*.mp4 2>/dev/null)" ]]; then
    echo "WARNING: $VAL_VIDEO_ROOT has no .mp4 files -- see this script's PREREQUISITE comment" >&2
    echo "block above. Every checkpoint will likely report n_eval=0 until validation clips are staged." >&2
fi

STEPS_ARGS=()
if [[ -n "$STEPS" ]]; then
    STEPS_ARGS=(--steps "$STEPS")
fi
VAL_EXCLUDE_ARGS=()
if [[ -f "$VAL_EXCLUDE_FLAGGED_JSON" ]]; then
    VAL_EXCLUDE_ARGS=(--val_exclude_flagged_json "$VAL_EXCLUDE_FLAGGED_JSON")
else
    echo "WARNING: $VAL_EXCLUDE_FLAGGED_JSON not found -- proceeding WITHOUT validation quality-screen exclusion." >&2
fi
WANDB_ARGS=()
if [[ -n "$WANDB_PROJECT" ]]; then
    WANDB_ARGS=(--wandb_project "$WANDB_PROJECT")
    [[ -n "$WANDB_ENTITY" ]] && WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    [[ -n "$WANDB_RUN_NAME" ]] && WANDB_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
    [[ -n "$WANDB_RESUME_RUN_ID" ]] && WANDB_ARGS+=(--wandb_resume_run_id "$WANDB_RESUME_RUN_ID")
fi

python "$SCRATCH/scripts/evaluate_stage0_latent_grounding.py" \
    --qwen_dir "$QWEN_DIR" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --val_metadata_dir "$VAL_METADATA_DIR" \
    --val_video_root "$VAL_VIDEO_ROOT" \
    --eval_max_examples "$EVAL_MAX_EXAMPLES" \
    "${STEPS_ARGS[@]}" \
    "${VAL_EXCLUDE_ARGS[@]}" \
    "${WANDB_ARGS[@]}"

echo "===== Validation-only evaluation completed — see $CHECKPOINT_ROOT/eval_only/ ====="
echo "  eval_metrics.jsonl        -- raw per-checkpoint numbers"
echo "  val_vs_train_loss.png     -- train vs val loss_total by step (THE graph)"
echo "  val_loss_components.png   -- val loss_ce / loss_latent split out separately"
echo "  train_vs_val_summary.jsonl -- per-checkpoint train/val/gap (the memorization-vs-generalization read)"
echo "End: $(date)"
