#!/bin/bash
#SBATCH --job-name=coconut_smoke
#SBATCH --partition=ada
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/coconut_smoke_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/coconut_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -uo pipefail
# Smoke test for vans_coconut.py: every arm trains 3 tiny curriculum stages, coconut + cot evaluate 2 samples.
T=/scratch/users/anirban/tamaghnam; S=$T/latentvans
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
PY=$T/envs/vans/bin/python
C="--vans_code $S/code/VANS --vans_dir $S/checkpoints/VANS --qwen_dir $S/checkpoints/Qwen2.5-VL-3B-Instruct"
O=$S/results/vans_smoke/coconut
for a in coconut pause; do
  echo "=== train $a"; $PY $S/scripts/vans_coconut.py --mode train --arm $a --data_dir $S/data/vans_train \
    --output_dir $O/$a --max_stage 2 --stage_steps 1 --grad_accum 1 --log_every 1 $C && echo "=== train $a OK" || echo "=== train $a FAILED"
done
for a in coconut cot; do
  echo "=== eval $a"; $PY $S/scripts/vans_coconut.py --mode eval --arm $a --ckpt $O/$a/final --max_stage 2 \
    --bench_dir $S/data/vans_eval --output_dir $O/eval --limit 2 --max_new_tokens 64 $C && echo "=== eval $a OK" || echo "=== eval $a FAILED"
done
