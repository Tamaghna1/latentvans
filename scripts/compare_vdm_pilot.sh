#!/bin/bash
#SBATCH --job-name=vdm_compare
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_compare_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_compare_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# Paired per-clip comparison of finished VDM-pilot arms (see compare_vdm_pilot.py).
#   ARM_DIRS="$SCRATCH/checkpoints/vdm_pilot/caption_vlm_quad_abs/step_3000 ..." sbatch compare_vdm_pilot.sh
# Default: every arm under checkpoints/vdm_pilot that has a step_3000.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
ARM_DIRS="${ARM_DIRS:-$(ls -d $SCRATCH/checkpoints/vdm_pilot/*/step_3000 | tr '\n' ' ')}"
WANDB_PROJECT="${WANDB_PROJECT:-latentvans-vdm-pilot}"

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)"
echo "ARM_DIRS=$ARM_DIRS"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

python "$SCRATCH/scripts/compare_vdm_pilot.py" --arm_dirs $ARM_DIRS \
    --wan_dir "$SCRATCH/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16" \
    --output_dir "$SCRATCH/checkpoints/vdm_pilot/comparison" \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"}
echo "Done: $(date)"
