#!/bin/bash
#SBATCH --job-name=anchor_gen
#SBATCH --partition=short,long,ada,medium
#SBATCH --exclude=cn1,cn2,cn7
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/anchor_gen_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/anchor_gen_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# Training-free scene anchoring (anchor_generate.py, plan.md P28 step 1), one shard per GPU, then CLIP/FVD metrics
# and the event judge (if downloaded).
#   NAME=<arm> CAPS=nocot|gt|<glob> [FFREP=1] [STRENGTH=0.5] [EVERY=4] [LIMIT=N: smoke, no scoring]
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
BENCH=$S/data/vans_eval
OUT=$S/results/vans_eval
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
PY=$T/envs/vans/bin/python
case ${CAPS:-nocot} in
  nocot) CAPS="$S/results/vans_coconut/coconut_nocot/captions_shard*.jsonl" ;;
esac
NG=$(nvidia-smi -L | wc -l)
echo "job ${SLURM_JOB_ID:-local} on $(hostname), $NG GPU(s), NAME=$NAME CAPS=$CAPS FFREP=${FFREP:-} STRENGTH=${STRENGTH:-1} EVERY=${EVERY:-1}"

PIDS=()
for ((k = 0; k < NG; k++)); do
  CUDA_VISIBLE_DEVICES=$k $PY $S/scripts/anchor_generate.py --name $NAME --captions "$CAPS" ${FFREP:+--ffrep} \
    --strength ${STRENGTH:-1} --every ${EVERY:-1} --bench_dir $BENCH --out_dir $OUT --vans_code $S/code/VANS \
    --vans_dir $S/checkpoints/VANS --wan_orig_dir $S/checkpoints/Wan2.1-T2V-1.3B-orig \
    --wan_diffusers_dir $S/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16 --shard $k --num_shards $NG ${LIMIT:+--limit $LIMIT} &
  PIDS+=($!)
done
for p in ${PIDS[@]}; do wait $p; done
[[ -n "${LIMIT:-}" ]] && { echo "smoke done: $(date)"; exit 0; }

$PY $S/scripts/vans_eval_metrics.py --bench_dir $BENCH --arm_dir $OUT/$NAME --i3d $S/checkpoints/i3d_torchscript.pt \
  --summary $OUT/summary.jsonl --wandb_project latentvans-vans-eval
J=$S/checkpoints/Qwen2.5-VL-7B-Instruct
[[ -f $J/config.json ]] && $PY $S/scripts/event_judge.py --bench_dir $BENCH --arm_dir $OUT/$NAME --results_dir $OUT \
  --qwen_dir $J --wandb_project latentvans-vans-eval
echo "Done: $(date)"
