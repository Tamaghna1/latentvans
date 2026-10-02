#!/bin/bash
#SBATCH --job-name=diagnose_stage0_grounding
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/diagnose_stage0_grounding_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/diagnose_stage0_grounding_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# Compares a Stage 0 checkpoint's held-out latent MSE against trivial baselines
# (predict the average target, copy the context frames) plus a retrieval test.
# See diagnose_stage0_grounding.py's docstring. Usage:
#   CHECKPOINT=$SCRATCH/checkpoints/<run>/step_<N> sbatch diagnose_stage0_grounding.sh
# Frames come from the v2 caches (correct TwiFF frame indexing, see twiff_frames.py).

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"

CHECKPOINT="${CHECKPOINT:-$SCRATCH/checkpoints/stage0_futurel150k_run1/step_2400}"
N_TRAIN="${N_TRAIN:-500}"
N_VAL="${N_VAL:-300}"
OUTPUT_JSON="${OUTPUT_JSON:-$CHECKPOINT/diagnostics/grounding.json}"
WANDB_PROJECT="${WANDB_PROJECT:-latentvans-stage0}"   # blank = no W&B logging

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)"
echo "CHECKPOINT=$CHECKPOINT"
nvidia-smi || true

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

python "$SCRATCH/scripts/diagnose_stage0_grounding.py" \
    --checkpoint "$CHECKPOINT" \
    --qwen_dir "$SCRATCH/checkpoints/Qwen2.5-VL-3B-Instruct" \
    --train_metadata_dir "$SCRATCH/data/future_l1_50k_metadata_full" \
    --train_video_root "$SCRATCH/data/panda70m_clips" \
    --train_frames_root "$SCRATCH/data/panda70m_frames_v2" \
    --val_metadata_dir "$SCRATCH/data/twiff_metadata_val" \
    --val_video_root "$SCRATCH/data/panda70m_clips_val" \
    --val_frames_root "$SCRATCH/data/panda70m_frames_val_v2" \
    --n_train "$N_TRAIN" --n_val "$N_VAL" \
    --output_json "$OUTPUT_JSON" \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"}

echo "Done: $(date)"
