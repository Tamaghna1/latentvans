#!/bin/bash
#SBATCH --job-name=vans_coconut
#SBATCH --partition=ada
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_coconut_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_coconut_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# Coconut-style continuous reasoning for the VANS VLM; see vans_coconut.py.
#   MODE=train ARM=cot|nocot|coconut|pause [EXTRA="..."]
#   MODE=eval  ARM=... CKPT=<dir>|init [NAME=...] [LIMIT=N]   (one shard per GPU, then caption scores)
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
PY=$T/envs/vans/bin/python
COMMON="--vans_code $S/code/VANS --vans_dir $S/checkpoints/VANS --qwen_dir $S/checkpoints/Qwen2.5-VL-3B-Instruct"

if [[ "$MODE" == train ]]; then
  $PY $S/scripts/vans_coconut.py --mode train --arm $ARM --data_dir $S/data/vans_train \
    --output_dir $S/checkpoints/vans_coconut/$ARM --wandb_project latentvans-coconut $COMMON ${EXTRA:-}
else
  NAME=${NAME:-coconut_${ARM}}
  NG=$(nvidia-smi -L | wc -l)
  PIDS=()
  for k in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$k $PY $S/scripts/vans_coconut.py --mode eval --arm $ARM --ckpt $CKPT --name $NAME \
      --bench_dir $S/data/vans_eval --output_dir $S/results/vans_coconut --shard $k --num_shards $NG \
      ${LIMIT:+--limit $LIMIT} $COMMON ${EXTRA:-} &
    PIDS+=($!)
  done
  for p in ${PIDS[@]}; do wait $p; done
  $PY - <<EOF
import glob, json, sys
sys.path.insert(0, "$S/scripts")
from vans_eval_metrics import text_metrics
bench = {json.loads(l)["sample_id"]: json.loads(l) for l in open("$S/data/vans_eval/benchmark.jsonl")}
caps = [json.loads(l) for f in glob.glob("$S/results/vans_coconut/$NAME/captions_shard*.jsonl") for l in open(f)]
m = text_metrics([c["caption"] for c in caps], [bench[c["sample_id"]]["gt_caption"] for c in caps])
m.update(n=len(caps), format_ok=sum(c["format_ok"] for c in caps) / max(len(caps), 1))
print("SCORES $NAME", json.dumps(m))
json.dump(m, open("$S/results/vans_coconut/$NAME/scores.json", "w"), indent=1)
with open("$S/results/vans_coconut/summary.jsonl", "a") as f:
    f.write(json.dumps({"name": "$NAME", **m}) + "\n")
EOF
fi
echo "Done: $(date)"
