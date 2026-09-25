#!/bin/bash
#SBATCH --job-name=phase2_smoke
#SBATCH --partition=ada
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/phase2_smoke_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/phase2_smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

# Runs phase2_pipeline_smoke_test.py -- see that file's module docstring for
# what it checks and why the two Wan checkpoint directories differ.
#
# Adjust #SBATCH --partition above to whichever GPU partition you want
# (a100 / ada / h200) -- all should have plenty of headroom for a batch-size-1
# smoke test with a 3B + 1.3B model pair, but a100 is picked as the default
# since it's the smallest GPU partition listed and passing here is a
# stronger signal for the eventual real training runs.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
CHECKPOINT_DIR="$SCRATCH/checkpoints"

QWEN_DIR="$CHECKPOINT_DIR/Qwen2.5-VL-3B-Instruct"
WAN_DIFFUSERS_DIR="$CHECKPOINT_DIR/Wan2.1-T2V-1.3B-Diffusers"

echo "===== Phase 2: Pipeline Smoke Test ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true   # full header (includes Driver Version / CUDA Version) -- needed to catch
                     # torch-vs-driver CUDA mismatches; --query-gpu alone suppresses this header

mkdir -p "$SCRATCH/logs"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

if [[ ! -d "$WAN_DIFFUSERS_DIR" ]]; then
    echo "Missing $WAN_DIFFUSERS_DIR -- run phase1b_download_wan_diffusers.sh first." >&2
    exit 1
fi

python -m pip install --upgrade 'transformers>=4.49.0' 'diffusers>=0.32.0' accelerate pillow numpy torchvision

# Expects phase2_pipeline_smoke_test.py copied to $SCRATCH alongside the
# other templates (same convention as phase1_data_staging.sh's siblings).
python "$SCRATCH/scripts/phase2_pipeline_smoke_test.py" \
    --qwen_dir "$QWEN_DIR" \
    --wan_diffusers_dir "$WAN_DIFFUSERS_DIR" \
    --dtype bf16 \
    --freeze_vlm

echo "===== Phase 2 completed successfully ====="
echo "End: $(date)"
