#!/bin/bash
#SBATCH --job-name=vdm_encode_texts
#SBATCH --partition=a100,ada,long,short
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_encode_texts_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_encode_texts_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# umT5-encode several caption variants in one job (see vdm_pilot_encode.py --what text).
#   TAGS="oracle full future" sbatch vdm_pilot_encode_texts.sh
# Tag X reads data/vdm_pilot/captions_X/{split}_vlm.pt and writes encoded/{split}_text_X.pt.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
TAGS="${TAGS:?set TAGS, e.g. TAGS=\"oracle full future\"}"
echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Start: $(date)  TAGS=$TAGS"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"
for tag in $TAGS; do
    python "$SCRATCH/scripts/vdm_pilot_encode.py" --what text \
        --wan_dir "$SCRATCH/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16" \
        --train_metadata_dir "$SCRATCH/data/future_l1_50k_metadata_full" \
        --train_rows_json "$SCRATCH/data/vdm_pilot/train_rows.json" \
        --train_video_root "$SCRATCH/data/panda70m_clips" \
        --val_metadata_dir "$SCRATCH/data/twiff_metadata_val" \
        --val_video_root "$SCRATCH/data/panda70m_clips_val" \
        --val_frames_root "$SCRATCH/data/panda70m_frames_val_v2" \
        --vlm_dir "$SCRATCH/data/vdm_pilot/captions_$tag" --text_tag "$tag" \
        --output_dir "$SCRATCH/data/vdm_pilot/encoded"
done
echo "Done: $(date)"
