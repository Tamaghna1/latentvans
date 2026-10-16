#!/bin/bash
#SBATCH --job-name=copy_base
#SBATCH --partition=short
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/copy_base_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/copy_base_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

# Scene-copy baselines (copy_last, copy_input; see copy_baselines.py), scored with the benchmark's metrics.
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
BENCH=$S/data/vans_eval
OUT=$S/results/vans_eval
export HF_HOME=$T/hf_cache
echo "job ${SLURM_JOB_ID:-local} on $(hostname)"

$T/envs/vans/bin/python $S/scripts/copy_baselines.py --bench_dir $BENCH --out_dir $OUT --ffmpeg $T/envs/latentvans/bin/ffmpeg
for A in copy_last copy_input; do
  $T/envs/vans/bin/python $S/scripts/vans_eval_metrics.py --bench_dir $BENCH --arm_dir $OUT/$A \
    --i3d $S/checkpoints/i3d_torchscript.pt --summary $OUT/summary.jsonl --wandb_project latentvans-vans-eval
done
echo "Done: $(date)"
