#!/bin/bash
#SBATCH --job-name=vans_eval_gen
#SBATCH --partition=a100
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_eval_gen_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_eval_gen_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail

# Generates VANS-benchmark answers for one arm on 2 GPUs (one shard per GPU), then scores it.
#   ARM=vans | vans_oracle                       released VANS model (envs/vans)
#   ARM=pilot NAME=<name> CKPT=<step dir> CAPTIONS=null|oracle|<glob>   a VDM pilot arm (envs/latentvans)
# Resubmitting resumes. LIMIT=N runs only the first N samples (smoke test).
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
BENCH=$S/data/vans_eval
OUT=$S/results/vans_eval
export HF_HOME=$T/hf_cache
mkdir -p $OUT
PIDS=()
LIM=${LIMIT:+--limit $LIMIT}

if [[ "$ARM" == pilot ]]; then
  PY=$T/envs/latentvans/bin/python
  for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$k $PY $S/scripts/vdm_pilot_generate.py --name $NAME --ckpt $CKPT --caption_source "$CAPTIONS" \
      --bench_dir $BENCH --out_dir $OUT --wan_dir $S/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16 \
      --shard $k --num_shards 2 $LIM &
    PIDS+=($!)
  done
  for p in ${PIDS[@]}; do wait $p; done
  ARM_DIR=$OUT/$NAME
else
  PY=$T/envs/vans/bin/python
  for k in 0 1; do
    CUDA_VISIBLE_DEVICES=$k $PY $S/scripts/vans_eval_generate.py --arm $ARM --bench_dir $BENCH --out_dir $OUT \
      --vans_code $S/code/VANS --vans_dir $S/checkpoints/VANS --wan_orig_dir $S/checkpoints/Wan2.1-T2V-1.3B-orig \
      --wan_diffusers_dir $S/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16 --qwen_dir $S/checkpoints/Qwen2.5-VL-3B-Instruct \
      --shard $k --num_shards 2 $LIM &
    PIDS+=($!)
  done
  for p in ${PIDS[@]}; do wait $p; done
  ARM_DIR=$OUT/$ARM
fi

$T/envs/vans/bin/python $S/scripts/vans_eval_metrics.py --bench_dir $BENCH --arm_dir $ARM_DIR \
  --i3d $S/checkpoints/i3d_torchscript.pt --summary $OUT/summary.jsonl --wandb_project latentvans-vans-eval
echo "Done: $(date)"
