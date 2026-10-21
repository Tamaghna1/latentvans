#!/bin/bash
#SBATCH --job-name=scene_sim
#SBATCH --partition=short,long,ada,medium
#SBATCH --exclude=cn1,cn2,cn7
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/scene_sim_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/scene_sim_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

# Scene retention vs the last input frame (scene_sim.py, plan.md P27 (b)) for answer-video arms in results/vans_eval.
#   ARMS="target_real vans ..." (default below; missing arms are skipped)  FORCE=1 (recompute)  LIMIT=N (smoke: first N rows,
# written to scene_smoke.json)
# Downloads facebook/dinov2-small (~90 MB) into hf_cache on first use.
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
PY=$T/envs/vans/bin/python
ARMS=${ARMS:-"target_real copy_last copy_last_vae copy_input vans_oracle vans captions_nocot captions_const anc_ffrep_e4 anc_sde05_e4 anc_sde05_gt_e4 anc_sde07_e4"}
echo "job ${SLURM_JOB_ID:-local} on $(hostname), arms: $ARMS"
$PY $S/scripts/scene_sim.py --bench_dir $S/data/vans_eval --results_dir $S/results/vans_eval --arms $ARMS \
  ${FORCE:+--force} ${LIMIT:+--limit $LIMIT} $([[ -z "${LIMIT:-}" ]] && echo --wandb_project latentvans-vans-eval)
echo "Done: $(date)"
