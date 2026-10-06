#!/bin/bash
# Driver for the caption-variant arms of the VDM pilot; run it inside tmux on the login node:
#   tmux new -d -s caption_variants "bash run_caption_variants_pipeline.sh"
# Submits each stage as soon as the account's submit limit (4 queued jobs) leaves room, chained
# with afterok dependencies so Slurm starts it when its inputs exist and a run slot (2) frees:
#   captions (full + future)  ->  umT5 encode (oracle full future)  ->
#   train [oracle, full] + train [future, oracle seed 1]  ->  paired comparison
# Each job also gets its own tmux session via tmux_job.sh --attach.
set -euo pipefail
cd /scratch/users/anirban/tamaghnam/latentvans/scripts
V=/scratch/users/anirban/tamaghnam/latentvans/checkpoints/vdm_pilot
MAX_SUBMITTED=4

submit() {  # submit <session-name> <env...> -- <sbatch args...>; echoes the job id
    local name="$1"; shift
    local envs=()
    while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done; shift
    while [[ $(squeue -u "$USER" -h | wc -l) -ge $MAX_SUBMITTED ]]; do sleep 60; done
    local id
    id=$(env "${envs[@]}" sbatch --parsable "$@")
    bash tmux_job.sh --attach "$name" "$id" >&2
    echo "$(date +%H:%M) submitted $name as $id" >&2
    echo "$id"
}

CAP=$(submit captions_gen -- vdm_pilot_captions.sh)
ENC=$(submit encode_texts TAGS="oracle full future" -- --dependency=afterok:$CAP vdm_pilot_encode_texts.sh)
T1=$(submit vdm_text_oracle_full ARMS="caption::0:oracle caption::0:full" -- --gres=gpu:2 --dependency=afterok:$ENC train_vdm_pilot_multi.sh)
T2=$(submit vdm_text_future_oracle1 ARMS="caption::0:future caption::1:oracle" -- --gres=gpu:2 --dependency=afterok:$ENC train_vdm_pilot_multi.sh)
DIRS="$V/caption_vlm_quad_abs/step_3000 $V/caption_vlm_quad_abs_seed1/step_3000 $V/null_vlm_quad_abs/step_3000 $V/null_vlm_quad_abs_seed1/step_3000 $V/caption_vlm_quad_abs_text-oracle/step_3000 $V/caption_vlm_quad_abs_text-oracle_seed1/step_3000 $V/caption_vlm_quad_abs_text-full/step_3000 $V/caption_vlm_quad_abs_text-future/step_3000"
CMP=$(submit compare_caption_variants ARM_DIRS="$DIRS" -- --dependency=afterok:$T1:$T2 --export=ALL,ARM_DIRS="$DIRS" --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vdm_compare_captions_%j.out compare_vdm_pilot.sh)
echo "$(date +%H:%M) all stages submitted: captions=$CAP encode=$ENC train=$T1,$T2 compare=$CMP"
