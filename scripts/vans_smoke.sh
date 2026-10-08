#!/bin/bash
#SBATCH --job-name=vans_smoke
#SBATCH --partition=a100
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_smoke_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vans_smoke_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -uo pipefail

# Runs every VANS eval / SFT / GRPO entry point on 2 samples / 2 steps, to catch wiring bugs before
# the real jobs. Each step is independent: a failure is reported and the next step still runs.
# STEPS="gen_vans gen_oracle gen_pilot metrics sft_vlm sft_vdm grpo1 grpo2" selects a subset.
T=/scratch/users/anirban/tamaghnam
S=$T/latentvans
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
V=$T/envs/vans/bin/python
L=$T/envs/latentvans/bin/python
SM=$S/results/vans_smoke
mkdir -p $SM
COMMON="--vans_code $S/code/VANS --wan_orig_dir $S/checkpoints/Wan2.1-T2V-1.3B-orig \
  --wan_diffusers_dir $S/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16 --qwen_dir $S/checkpoints/Qwen2.5-VL-3B-Instruct"
STEPS=${STEPS:-"gen_vans gen_oracle gen_pilot metrics sft_vlm sft_vdm grpo1 grpo2"}
declare -A RESULT

run() {  # run <name> <cmd...>
  local name=$1; shift
  [[ " $STEPS " == *" $name "* ]] || return 0
  echo "=================== $name: $(date +%T)"
  "$@" && RESULT[$name]=OK || RESULT[$name]="FAILED ($?)"
  echo "=================== $name: ${RESULT[$name]} $(date +%T)"
}

run gen_vans $V $S/scripts/vans_eval_generate.py --arm vans --bench_dir $S/data/vans_eval --out_dir $SM --limit 2 \
  --vans_dir $S/checkpoints/VANS $COMMON
run gen_oracle $V $S/scripts/vans_eval_generate.py --arm vans_oracle --bench_dir $S/data/vans_eval --out_dir $SM \
  --limit 2 --vans_dir $S/checkpoints/VANS $COMMON
run gen_pilot $L $S/scripts/vdm_pilot_generate.py --name pilot_oracle --ckpt $S/checkpoints/vdm_pilot/caption_vlm_quad_abs_text-oracle/step_3000 \
  --caption_source oracle --bench_dir $S/data/vans_eval --out_dir $SM --limit 2 \
  --wan_dir $S/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16
for a in vans vans_oracle pilot_oracle; do
  run metrics $V $S/scripts/vans_eval_metrics.py --bench_dir $S/data/vans_eval --arm_dir $SM/$a \
    --i3d $S/checkpoints/i3d_torchscript.pt --summary $SM/summary.jsonl
done
run sft_vlm $V $S/scripts/vans_sft.py --stage vlm --data_dir $S/data/vans_train --output_dir $SM/sft_vlm \
  --max_steps 3 --log_every 1 --save_every 3 $COMMON
run sft_vdm $V $S/scripts/vans_sft.py --stage vdm --data_dir $S/data/vans_train --output_dir $SM/sft_vdm \
  --max_steps 3 --log_every 1 --save_every 1000 $COMMON
run grpo1 $V $S/scripts/vans_grpo.py --stage 1 --data_dir $S/data/vans_train --output_dir $SM/grpo1 \
  --vlm_lora $S/checkpoints/VANS/VANS_mllm.safetensors --dit $S/checkpoints/VANS/VANS_vlm.safetensors \
  --num_prompts 4 --max_steps 2 --group_size 2 --reward_steps 4 --max_new_tokens 256 --save_every 2 $COMMON
run grpo2 $V $S/scripts/vans_grpo.py --stage 2 --data_dir $S/data/vans_train --output_dir $SM/grpo2 \
  --vlm_lora $S/checkpoints/VANS/VANS_mllm.safetensors --dit $S/checkpoints/VANS/VANS_vlm.safetensors \
  --num_prompts 8 --max_steps 2 --group_size 2 --sde_steps 4 --timesteps_per_update 2 --anchor_rouge 0.0 \
  --max_new_tokens 256 --save_every 1000 $COMMON

echo "SUMMARY"; for k in "${!RESULT[@]}"; do echo "  $k: ${RESULT[$k]}"; done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
