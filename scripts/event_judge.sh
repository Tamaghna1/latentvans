#!/bin/bash
#SBATCH --job-name=event_judge
#SBATCH --partition=short,long,ada,medium
#SBATCH --exclude=cn1,cn2,cn7
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/event_judge_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/event_judge_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# Event-centric judge (event_judge.py, plan.md P27) over answer-video arms in results/vans_eval, one after another.
#   ARMS="target_real vans ..." (default below)  JUDGE=Qwen2.5-VL-7B-Instruct  LIMIT=N (smoke: first N rows)
# Downloads the judge into checkpoints/ on first use.
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
BENCH=$S/data/vans_eval
OUT=$S/results/vans_eval
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
PY=$T/envs/vans/bin/python
JUDGE=${JUDGE:-Qwen2.5-VL-7B-Instruct}
ARMS=${ARMS:-"target_real copy_last copy_input vans_oracle vans captions_nocot captions_const"}
echo "job ${SLURM_JOB_ID:-local} on $(hostname), judge $JUDGE, arms: $ARMS"

if [[ ! -f $S/checkpoints/$JUDGE/config.json ]]; then
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/$JUDGE', local_dir='$S/checkpoints/$JUDGE')"
fi
for A in $ARMS; do
  [[ $A == target_real ]] && D=target_real || D=$OUT/$A
  $PY $S/scripts/event_judge.py --bench_dir $BENCH --arm_dir $D --results_dir $OUT --qwen_dir $S/checkpoints/$JUDGE \
    ${LIMIT:+--limit $LIMIT --tag ${JUDGE}_smoke} $([[ -z "${LIMIT:-}" ]] && echo --wandb_project latentvans-vans-eval)
done
echo "Done: $(date)"
