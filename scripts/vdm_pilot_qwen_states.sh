#!/bin/bash
#SBATCH --job-name=vdm_qwen_states
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_qwen_states_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_qwen_states_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
set -euo pipefail

# Qwen hidden states over its own generated answers for the VDM pilot "qwen" arm
# (see vdm_pilot_qwen_states.py).

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
CHECKPOINT="${CHECKPOINT:-$SCRATCH/checkpoints/stage0_v2_quad_abs/step_6000}"
VLM_DIR="${VLM_DIR:-$SCRATCH/data/vdm_pilot/vlm_quad_abs}"
EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

python "$SCRATCH/scripts/vdm_pilot_qwen_states.py" \
    --checkpoint "$CHECKPOINT" \
    --qwen_dir "$SCRATCH/checkpoints/Qwen2.5-VL-3B-Instruct" \
    --vlm_dir "$VLM_DIR" \
    --train_metadata_dir "$SCRATCH/data/future_l1_50k_metadata_full" \
    --train_frames_root "$SCRATCH/data/panda70m_frames_v2" \
    --val_metadata_dir "$SCRATCH/data/twiff_metadata_val" \
    --val_frames_root "$SCRATCH/data/panda70m_frames_val_v2" "${EXTRA[@]}"
echo "Done: $(date)"
