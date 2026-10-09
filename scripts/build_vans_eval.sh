#!/bin/bash
#SBATCH --job-name=build_vans_eval
#SBATCH --partition=long
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/build_vans_eval_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/build_vans_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
set -euo pipefail

# Rebuilds the procedural half of the VANS benchmark (200 YouCook2-val + 200 COIN-test rows)
# into data/vans_eval. Network-bound, no GPU. Resubmitting resumes where it stopped.
# SPLIT=train OUT=data/vans_train PER_SOURCE=2000 ROWS_PER_VIDEO=3 builds SFT/RL training data instead.
SCRATCH=/scratch/users/anirban/tamaghnam/latentvans
export PATH=/scratch/users/anirban/tamaghnam/envs/latentvans/bin:$PATH
cd $SCRATCH/scripts
# NUM_SHARDS>1 runs that many builders in parallel inside this job (separate video subsets)
N=${NUM_SHARDS:-1}
PIDS=()
for k in $(seq 0 $((N - 1))); do
python build_vans_eval.py \
  --meta_dir $SCRATCH/data/vans_meta \
  --out_dir $SCRATCH/${OUT:-data/vans_eval} \
  --per_source ${PER_SOURCE:-200} \
  --split ${SPLIT:-eval} --rows_per_video ${ROWS_PER_VIDEO:-1} \
  --shard $k --num_shards $N \
  --cookies $SCRATCH/cookies.txt > $SCRATCH/logs/build_vans_${SPLIT:-eval}_${SLURM_JOB_ID}_shard$k.log 2>&1 &
PIDS+=($!)
done
for p in ${PIDS[@]}; do wait $p; done
echo "Done: $(date)"
