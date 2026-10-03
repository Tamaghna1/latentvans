#!/bin/bash
#SBATCH --job-name=vdm_pilot_vlm
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_pilot_vlm_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_pilot_vlm_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
set -euo pipefail

# Latents (+ captions unless NO_CAPTIONS=1) from one Stage 0 checkpoint for the VDM pilot.
# See vdm_pilot_vlm_outputs.py. Resumable: rerun with the same OUTPUT_DIR to continue.
#   CHECKPOINT=$SCRATCH/checkpoints/stage0_v2_quad_abs/step_6000 OUTPUT_DIR=$SCRATCH/data/vdm_pilot/vlm_quad_abs sbatch vdm_pilot_vlm_outputs.sh

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a Stage 0 step_N dir}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
EXTRA=()
[[ "${NO_CAPTIONS:-0}" == "1" ]] && EXTRA+=(--no_captions)
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)  CHECKPOINT=$CHECKPOINT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

python "$SCRATCH/scripts/vdm_pilot_vlm_outputs.py" \
    --checkpoint "$CHECKPOINT" \
    --qwen_dir "$SCRATCH/checkpoints/Qwen2.5-VL-3B-Instruct" \
    --train_metadata_dir "$SCRATCH/data/future_l1_50k_metadata_full" \
    --train_rows_json "$SCRATCH/data/vdm_pilot/train_rows.json" \
    --train_frames_root "$SCRATCH/data/panda70m_frames_v2" \
    --val_metadata_dir "$SCRATCH/data/twiff_metadata_val" \
    --val_frames_root "$SCRATCH/data/panda70m_frames_val_v2" \
    --output_dir "$OUTPUT_DIR" "${EXTRA[@]}"
echo "Done: $(date)"
