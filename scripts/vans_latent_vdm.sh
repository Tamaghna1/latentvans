#!/bin/bash
#SBATCH --job-name=vans_latent
#SBATCH --partition=a100
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_latent_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_latent_%j.err
#SBATCH --time=23:50:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail

# Core-claim experiment (caption vs caption + end-to-end latent, VANS setting); see vans_latent_vdm.py.
#   MODE=train ARM=caption|latent [STEPS=4000] [EXTRA="..."]
#   MODE=generate ARM=caption|latent CKPT=<step dir> CAPTIONS=vans|oracle   (uses --gres=gpu:2 if given)
#   MODE=cache   pre-encode training rows into data/vans_train_cache, one shard per GPU
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
PY=$T/envs/vans/bin/python
COMMON="--vans_code $S/code/VANS --vans_dir $S/checkpoints/VANS --wan_orig_dir $S/checkpoints/Wan2.1-T2V-1.3B-orig \
  --wan_diffusers_dir $S/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16 --qwen_dir $S/checkpoints/Qwen2.5-VL-3B-Instruct"

if [[ "$MODE" == cache ]]; then
  NG=$(nvidia-smi -L | wc -l)
  PIDS=()
  for k in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$k $PY $S/scripts/vans_latent_vdm.py --mode cache --data_dir $S/data/vans_train \
      --cache_dir $S/data/vans_train_cache --output_dir $S/data/vans_train_cache --shard $k --num_shards $NG $COMMON &
    PIDS+=($!)
  done
  for p in ${PIDS[@]}; do wait $p; done
elif [[ "$MODE" == train ]]; then
  $PY $S/scripts/vans_latent_vdm.py --mode train --arm $ARM --data_dir $S/data/vans_train \
    --output_dir $S/checkpoints/vans_latent/$ARM --cache_dir $S/data/vans_train_cache \
    --max_steps ${STEPS:-4000} --wandb_project latentvans-vans-latent $COMMON ${EXTRA:-}
else
  NAME=latent_${ARM}_${CAPTIONS}
  NG=$(nvidia-smi -L | wc -l)
  PIDS=()
  for k in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$k $PY $S/scripts/vans_latent_vdm.py --mode generate --arm $ARM --ckpt $CKPT \
      --captions $CAPTIONS --vans_captions "$S/results/vans_eval/vans/captions_shard*.jsonl" --name $NAME \
      --bench_dir $S/data/vans_eval --output_dir $S/results/vans_eval --shard $k --num_shards $NG \
      ${LIMIT:+--limit $LIMIT} $COMMON &
    PIDS+=($!)
  done
  for p in ${PIDS[@]}; do wait $p; done
  $PY $S/scripts/vans_eval_metrics.py --bench_dir $S/data/vans_eval --arm_dir $S/results/vans_eval/$NAME \
    --i3d $S/checkpoints/i3d_torchscript.pt --summary $S/results/vans_eval/summary.jsonl \
    --wandb_project latentvans-vans-eval
fi
echo "Done: $(date)"
