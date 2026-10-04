#!/bin/bash
#SBATCH --job-name=vdm_multi
#SBATCH --partition=a100,ada,long
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_multi_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_multi_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
set -uo pipefail

# Several VDM-pilot arms in ONE Slurm job, one arm per GPU, so the account's 2-running-job
# limit isn't the bottleneck. Each arm runs train_vdm_pilot.py exactly as train_vdm_pilot.sh
# would, with its own output dir, log file and W&B run.
#
#   ARMS="qwen:last:0 qwen:mid:0" bash tmux_job.sh vdm_qwen sbatch --gres=gpu:2 train_vdm_pilot_multi.sh
#   ARMS="caption::1 null::1 latent::1 both::1" bash tmux_job.sh vdm_seed1 sbatch --gres=gpu:4 train_vdm_pilot_multi.sh
#
# ARMS entries are cond:qwen_layer:seed (qwen_layer only for cond=qwen). Request as many GPUs
# as arms. Partition list: a100 (80 GB), ada (ADA6000 48 GB), long (A6000 48 GB) -- Slurm
# starts the job wherever GPUs are free first; --mem=80G fits all three caps.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
ARMS="${ARMS:?set ARMS, e.g. ARMS=\"qwen:last:0 qwen:mid:0\"}"
VLM_DIR="${VLM_DIR:-$SCRATCH/data/vdm_pilot/vlm_quad_abs}"
MAX_STEPS="${MAX_STEPS:-3000}"
WANDB_PROJECT="${WANDB_PROJECT:-latentvans-vdm-pilot}"

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)  Partition: ${SLURM_JOB_PARTITION:-?}  Start: $(date)"
echo "ARMS=$ARMS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
i=0; pids=()
for arm in $ARMS; do
    IFS=':' read -r cond layer seed <<< "$arm"
    seed="${seed:-0}"
    tag="${cond}_$(basename "$VLM_DIR")"
    extra=(--seed "$seed")
    if [[ "$cond" == "qwen" ]]; then tag="${tag}_${layer:-last}"; extra+=(--qwen_layer "${layer:-last}"); fi
    [[ "$seed" != "0" ]] && tag="${tag}_seed${seed}"
    gpu="${GPUS[$i]:-}"
    if [[ -z "$gpu" ]]; then echo "more arms than GPUs (${#GPUS[@]}); skipping $arm"; continue; fi
    log="$SCRATCH/logs/vdm_multi_${SLURM_JOB_ID:-local}_${tag}.log"
    echo "arm $arm -> GPU $gpu, output checkpoints/vdm_pilot/$tag, log $log"
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRATCH/scripts/train_vdm_pilot.py" --cond "$cond" \
        --wan_dir "$SCRATCH/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16" \
        --encoded_dir "$SCRATCH/data/vdm_pilot/encoded" --vlm_dir "$VLM_DIR" \
        --output_dir "$SCRATCH/checkpoints/vdm_pilot/$tag" --max_steps "$MAX_STEPS" \
        ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} --wandb_run_name "vdm-$tag" \
        "${extra[@]}" > "$log" 2>&1 &
    pids+=($!); i=$((i + 1))
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "All arms finished (status $status): $(date)"
exit $status
