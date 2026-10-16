#!/bin/bash
#SBATCH --job-name=vae_copy
#SBATCH --partition=short
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vae_copy_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vae_copy_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
set -euo pipefail

# Sharpness / compression controls for the copy baselines (copy_last_q5, copy_last_vae, copy_input_vae; see
# vae_copy_baselines.py), scored with the benchmark's metrics. LIMIT=N: first N rows only, no scoring (smoke).
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
BENCH=$S/data/vans_eval
OUT=$S/results/vans_eval
export HF_HOME=$T/hf_cache
echo "job ${SLURM_JOB_ID:-local} on $(hostname)"

$T/envs/vans/bin/python $S/scripts/vae_copy_baselines.py --bench_dir $BENCH --out_dir $OUT \
  --vans_code $S/code/VANS --wan_orig_dir $S/checkpoints/Wan2.1-T2V-1.3B-orig ${LIMIT:+--limit $LIMIT}
if [[ -z "${LIMIT:-}" ]]; then
  for A in copy_last_q5 copy_last_vae copy_input_vae; do
    $T/envs/vans/bin/python $S/scripts/vans_eval_metrics.py --bench_dir $BENCH --arm_dir $OUT/$A \
      --i3d $S/checkpoints/i3d_torchscript.pt --summary $OUT/summary.jsonl --wandb_project latentvans-vans-eval
  done
fi
echo "Done: $(date)"
