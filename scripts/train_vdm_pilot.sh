#!/bin/bash
#SBATCH --job-name=vdm_pilot
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_pilot_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_pilot_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# One arm of the VDM conditioning pilot (see train_vdm_pilot.py). Submit via tmux_job.sh:
#   bash tmux_job.sh vdm_caption COND=caption sbatch train_vdm_pilot.sh
# COND = null | caption | latent | both. VLM_DIR picks whose latents (default: quad_abs Stage 0).

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
COND="${COND:?set COND=null|caption|latent|both}"
VLM_DIR="${VLM_DIR:-$SCRATCH/data/vdm_pilot/vlm_quad_abs}"
VLM_TAG="$(basename "$VLM_DIR")"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/checkpoints/vdm_pilot/${COND}_${VLM_TAG}}"
MAX_STEPS="${MAX_STEPS:-3000}"
WANDB_PROJECT="${WANDB_PROJECT:-latentvans-vdm-pilot}"

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)  COND=$COND VLM_DIR=$VLM_DIR"
nvidia-smi || true
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

python "$SCRATCH/scripts/train_vdm_pilot.py" --cond "$COND" \
    --wan_dir "$SCRATCH/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16" \
    --encoded_dir "$SCRATCH/data/vdm_pilot/encoded" \
    --vlm_dir "$VLM_DIR" \
    --output_dir "$OUTPUT_DIR" --max_steps "$MAX_STEPS" \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
    --wandb_run_name "vdm-${COND}-${VLM_TAG}" ${EXTRA_ARGS:-}
echo "Done: $(date)"
