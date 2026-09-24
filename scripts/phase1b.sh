#!/bin/bash
#SBATCH --job-name=phase1b_wan_diffusers
#SBATCH --partition=medium
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/phase1b_wan_diffusers_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/phase1b_wan_diffusers_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# ---------------------------------------------------------------------------
# WHY THIS SCRIPT EXISTS (read before running):
#
# phase1_data_staging.sh downloads "Wan-AI/Wan2.1-T2V-1.3B" (no suffix).
# That repo ships Wan's ORIGINAL checkpoint format:
#   - config.json has "_class_name": "WanModel" with keys like dim/in_dim/
#     out_dim/num_heads/text_len -- this is the schema used by the official
#     Wan2.1 GitHub repo's own wan.modules.model.WanModel class.
#   - VAE and T5 text encoder are raw .pth pickle files
#     (Wan2.1_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth), not diffusers
#     component folders.
#
# The Phase 2 smoke test (and, going forward, the diffusers-based training
# code in this project) uses HuggingFace `diffusers` classes directly
# (WanTransformer3DModel, AutoencoderKLWan) because they're the
# stable/documented API and match what we'll want for LoRA-style adaptation
# in Stage 2. Those classes expect the DIFFUSERS-format repo instead:
#   "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# which has transformer/, vae/, text_encoder/, tokenizer/, scheduler/
# subfolders with diffusers-native config.json schemas (attention_head_dim,
# text_dim=4096, etc).
#
# The two checkpoints are NOT interchangeable as drop-in files -- loading
# the native-format repo with diffusers.WanTransformer3DModel.from_pretrained
# will fail (unknown class name / missing keys). This script fetches the
# diffusers-format checkpoint as an ADDITION, without touching the
# already-confirmed-working phase1_data_staging.sh or its output.
#
# If a later stage ends up needing Wan's own repo/scripts instead of
# diffusers (e.g. if VANS's released training code hard-depends on
# wan.modules.model.WanModel), keep both checkpoints on scratch -- they're
# for two different code paths, not two versions of the same thing.
# ---------------------------------------------------------------------------

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
CHECKPOINT_DIR="$SCRATCH/checkpoints"
WAN_DIFFUSERS_DIR="$CHECKPOINT_DIR/Wan2.1-T2V-1.3B-Diffusers"

echo "===== Phase 1b: Wan2.1-T2V-1.3B (diffusers format) download ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"

mkdir -p "$CHECKPOINT_DIR" "$SCRATCH/logs"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

python -m pip install --upgrade huggingface_hub

echo "Downloading Wan-AI/Wan2.1-T2V-1.3B-Diffusers..."
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "$WAN_DIFFUSERS_DIR"

echo "===== Phase 1b completed successfully ====="
echo "Diffusers-format checkpoint at: $WAN_DIFFUSERS_DIR"
echo "End: $(date)"
