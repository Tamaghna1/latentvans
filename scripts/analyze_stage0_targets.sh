#!/bin/bash
#SBATCH --job-name=analyze_stage0_targets
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/analyze_stage0_targets_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/analyze_stage0_targets_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail

# Measures the Stage 0 target variants in stage0_targets.py on the v2 (correctly
# indexed) frame caches and writes their normalization stats -- see
# analyze_stage0_targets.py's docstring. Run after extract_stage0_frames.sh has
# produced panda70m_frames_v2 and panda70m_frames_val_v2.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"

N_TRAIN="${N_TRAIN:-3000}"
N_VAL="${N_VAL:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/data/stage0_targets_v2}"

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)"
nvidia-smi || true
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

python "$SCRATCH/scripts/analyze_stage0_targets.py" \
    --qwen_dir "$SCRATCH/checkpoints/Qwen2.5-VL-3B-Instruct" \
    --train_metadata_dir "$SCRATCH/data/future_l1_50k_metadata_full" \
    --train_video_root "$SCRATCH/data/panda70m_clips" \
    --train_frames_root "$SCRATCH/data/panda70m_frames_v2" \
    --val_metadata_dir "$SCRATCH/data/twiff_metadata_val" \
    --val_video_root "$SCRATCH/data/panda70m_clips_val" \
    --val_frames_root "$SCRATCH/data/panda70m_frames_val_v2" \
    --n_train "$N_TRAIN" --n_val "$N_VAL" \
    --output_dir "$OUTPUT_DIR"

echo "Done: $(date)"
