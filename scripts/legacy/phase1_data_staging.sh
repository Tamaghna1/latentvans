#!/bin/bash
#SBATCH --job-name=phase1_data
#SBATCH --partition=medium
#SBATCH --output=logs/phase1_data_%j.out
#SBATCH --error=logs/phase1_data_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
CODE_DIR="$SCRATCH/code"
CHECKPOINT_DIR="$SCRATCH/checkpoints"
DATA_DIR="$SCRATCH/data"
VANS_DIR="$CODE_DIR/VANS"
TWIFF_DIR="$CODE_DIR/TwiFF"
QWEN_DIR="$CHECKPOINT_DIR/Qwen2.5-VL-3B-Instruct"
WAN_DIR="$CHECKPOINT_DIR/Wan2.1-T2V-1.3B"
METADATA_DIR="$DATA_DIR/twiff_metadata_2000"

echo "===== Phase 1: Data Staging ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"

mkdir -p "$CODE_DIR" "$CHECKPOINT_DIR" "$DATA_DIR" "$SCRATCH/logs"
cd "$SCRATCH"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "Python: $(command -v python)"
python --version

python -m pip install --upgrade pip
python -m pip install --upgrade huggingface_hub datasets

if [[ ! -d "$VANS_DIR/.git" ]]; then
    if [[ -e "$VANS_DIR" ]]; then
        echo "Refusing to overwrite existing non-Git path: $VANS_DIR" >&2
        exit 1
    fi
    git clone https://github.com/AILab-CVC/VANS.git "$VANS_DIR"
else
    echo "VANS already exists; skipping clone."
fi

if [[ ! -d "$TWIFF_DIR/.git" ]]; then
    if [[ -e "$TWIFF_DIR" ]]; then
        echo "Refusing to overwrite existing non-Git path: $TWIFF_DIR" >&2
        exit 1
    fi
    git clone https://github.com/IntMeGroup/TwiFF.git "$TWIFF_DIR"
else
    echo "TwiFF already exists; skipping clone."
fi

echo "Downloading Qwen/Qwen2.5-VL-3B-Instruct..."
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$QWEN_DIR"

echo "Downloading Wan-AI/Wan2.1-T2V-1.3B..."
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$WAN_DIR"

echo "Streaming the first 2,000 TwiFF rows..."
export METADATA_DIR
python -c '
import os
from itertools import islice

from datasets import Dataset, load_dataset

output_dir = os.environ["METADATA_DIR"]
stream = load_dataset("Liu-Junhua/TwiFF-2.7M", split="train", streaming=True)
rows = list(islice(stream, 2000))
if len(rows) != 2000:
    raise RuntimeError(f"Expected 2,000 rows, received {len(rows)}")
Dataset.from_list(rows).save_to_disk(output_dir)
print(f"Saved {len(rows)} rows to {output_dir}")
'

echo "===== Phase 1 completed successfully ====="
echo "End: $(date)"
