#!/bin/bash
#SBATCH --job-name=vdm_captions
#SBATCH --partition=a100,ada,long,short
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_captions_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_captions_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
set -uo pipefail

# Generate the "full" (GPU 0) and "future" (GPU 1) caption variants in parallel for the VDM
# pilot (see vdm_pilot_captions.py). The "oracle" variant needs no GPU:
#   python vdm_pilot_captions.py --mode oracle ... (run once on the login node)

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
COMMON=(--keys_from "$SCRATCH/data/vdm_pilot/vlm_quad_abs"
        --qwen_dir "$SCRATCH/checkpoints/Qwen2.5-VL-3B-Instruct"
        --train_metadata_dir "$SCRATCH/data/future_l1_50k_metadata_full"
        --train_frames_root "$SCRATCH/data/panda70m_frames_v2"
        --val_metadata_dir "$SCRATCH/data/twiff_metadata_val"
        --val_frames_root "$SCRATCH/data/panda70m_frames_val_v2")
[[ -n "${LIMIT:-}" ]] && COMMON+=(--limit "$LIMIT")

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-?}  Start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" python "$SCRATCH/scripts/vdm_pilot_captions.py" --mode full \
    --checkpoint "$SCRATCH/checkpoints/stage0_v2_quad_abs/step_6000" "${COMMON[@]}" \
    --output_dir "$SCRATCH/data/vdm_pilot/captions_full" > "$SCRATCH/logs/vdm_captions_${SLURM_JOB_ID:-local}_full.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES="${GPUS[1]:-${GPUS[0]}}" python "$SCRATCH/scripts/vdm_pilot_captions.py" --mode future \
    "${COMMON[@]}" --output_dir "$SCRATCH/data/vdm_pilot/captions_future" \
    > "$SCRATCH/logs/vdm_captions_${SLURM_JOB_ID:-local}_future.log" 2>&1 &
P2=$!
status=0
wait $P1 || status=1
wait $P2 || status=1
echo "Done (status $status): $(date)"
exit $status
