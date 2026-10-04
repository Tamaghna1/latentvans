#!/bin/bash
#SBATCH --job-name=vdm_pilot_encode
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_pilot_encode_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_pilot_encode_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
set -euo pipefail

# Precompute VDM-pilot inputs (see vdm_pilot_encode.py):
#   WHAT=video sbatch vdm_pilot_encode.sh
#   WHAT=text VLM_DIR=$SCRATCH/data/vdm_pilot/vlm_quad_abs sbatch vdm_pilot_encode.sh

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
WHAT="${WHAT:?set WHAT=video or WHAT=text}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/data/vdm_pilot/encoded}"
EXTRA=()
[[ -n "${VLM_DIR:-}" ]] && EXTRA+=(--vlm_dir "$VLM_DIR")
[[ -n "${TEXT_TAG:-}" ]] && EXTRA+=(--text_tag "$TEXT_TAG")
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)  WHAT=$WHAT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

python "$SCRATCH/scripts/vdm_pilot_encode.py" --what "$WHAT" \
    --wan_dir "$SCRATCH/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16" \
    --train_metadata_dir "$SCRATCH/data/future_l1_50k_metadata_full" \
    --train_rows_json "$SCRATCH/data/vdm_pilot/train_rows.json" \
    --train_video_root "$SCRATCH/data/panda70m_clips" \
    --val_metadata_dir "$SCRATCH/data/twiff_metadata_val" \
    --val_video_root "$SCRATCH/data/panda70m_clips_val" \
    --val_frames_root "$SCRATCH/data/panda70m_frames_val_v2" \
    --output_dir "$OUTPUT_DIR" "${EXTRA[@]}"
echo "Done: $(date)"
