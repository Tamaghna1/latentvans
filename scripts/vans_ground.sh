#!/bin/bash
#SBATCH --job-name=vans_ground
#SBATCH --partition=ada
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_ground_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_ground_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# Visually grounded captioning for the VANS VLM; see ground_objects.py and vans_ground.py.
#   MODE=objects [LIMIT=N]                          grounding pre-pass over vans_train + vans_eval (one shard per GPU)
#   MODE=train ARM=ground_text|ground_zoom|ground_think [OUTDIR=...] [EXTRA="..."]
#   MODE=eval  ARM=... CKPT=<dir> [NAME=...] [LIMIT=N]   (one shard per GPU, then caption scores)
#   MODE=smoke                                      3 train steps + 3 eval rows for every arm
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
PY=$T/envs/vans/bin/python
COMMON="--vans_code $S/code/VANS --vans_dir $S/checkpoints/VANS --qwen_dir $S/checkpoints/Qwen2.5-VL-3B-Instruct"
NG=$(nvidia-smi -L | wc -l)
echo "job ${SLURM_JOB_ID:-?} on $(hostname), $NG GPU(s), MODE=$MODE ARM=${ARM:-}"

score() {  # caption scores for results/vans_ground/$1
  $PY - <<EOF
import glob, json, sys
sys.path.insert(0, "$S/scripts")
from vans_eval_metrics import text_metrics
bench = {json.loads(l)["sample_id"]: json.loads(l) for l in open("$S/data/vans_eval/benchmark.jsonl")}
caps = [json.loads(l) for f in glob.glob("$S/results/vans_ground/$1/captions_shard*.jsonl") for l in open(f)]
m = text_metrics([c["caption"] for c in caps], [bench[c["sample_id"]]["gt_caption"] for c in caps])
m.update(n=len(caps), format_ok=sum(c["format_ok"] for c in caps) / max(len(caps), 1))
print("SCORES $1", json.dumps(m))
json.dump(m, open("$S/results/vans_ground/$1/scores.json", "w"), indent=1)
with open("$S/results/vans_ground/summary.jsonl", "a") as f:
    f.write(json.dumps({"name": "$1", **m}) + "\n")
EOF
}

if [[ "$MODE" == objects ]]; then
  PIDS=()
  for k in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$k $PY $S/scripts/ground_objects.py --data_dirs $S/data/vans_eval $S/data/vans_train \
      --qwen_dir $S/checkpoints/Qwen2.5-VL-3B-Instruct --shard $k --num_shards $NG ${LIMIT:+--limit $LIMIT} &
    PIDS+=($!)
  done
  for p in ${PIDS[@]}; do wait $p; done
  for d in vans_eval vans_train; do
    echo "$d: $(cat $S/data/$d/ground/objects_shard*.jsonl | wc -l) rows grounded, $(du -sh $S/data/$d/ground | cut -f1)"
  done
elif [[ "$MODE" == train ]]; then
  $PY $S/scripts/vans_ground.py --mode train --arm $ARM --data_dir $S/data/vans_train \
    --output_dir ${OUTDIR:-$S/checkpoints/vans_ground/$ARM} --wandb_project latentvans-grounding \
    --wandb_run_name ground-$ARM-${SLURM_JOB_ID:-local} $COMMON ${EXTRA:-}
elif [[ "$MODE" == eval ]]; then
  NAME=${NAME:-$ARM}
  PIDS=()
  for k in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$k $PY $S/scripts/vans_ground.py --mode eval --arm $ARM --ckpt $CKPT --name $NAME \
      --bench_dir $S/data/vans_eval --output_dir $S/results/vans_ground --shard $k --num_shards $NG \
      ${LIMIT:+--limit $LIMIT} $COMMON ${EXTRA:-} &
    PIDS+=($!)
  done
  for p in ${PIDS[@]}; do wait $p; done
  score $NAME
elif [[ "$MODE" == smoke ]]; then
  for A in ground_text ground_zoom ground_think; do
    O=$S/checkpoints/vans_ground_smoke/$A
    $PY $S/scripts/vans_ground.py --mode train --arm $A --data_dir $S/data/vans_train --output_dir $O \
      --stage_steps 3 --max_stage 0 --grad_accum 2 --log_every 1 $COMMON
    $PY $S/scripts/vans_ground.py --mode eval --arm $A --ckpt $O/final --name smoke_$A --limit 3 \
      --bench_dir $S/data/vans_eval --output_dir $S/results/vans_ground_smoke $COMMON
    cat $S/results/vans_ground_smoke/smoke_$A/captions_shard0.jsonl
  done
  rm -rf $S/checkpoints/vans_ground_smoke $S/results/vans_ground_smoke
fi
echo "Done: $(date)"
