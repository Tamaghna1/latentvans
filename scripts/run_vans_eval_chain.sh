#!/bin/bash
# Full VANS-benchmark evaluation: 2 released-VANS arms, then 3 VDM-pilot arms (pilot "caption" renders
# the VANS VLM captions, so it waits for the vans arm). Respects the 4-submitted-job limit by waiting.
set -euo pipefail
cd /scratch/users/anirban/tamaghnam/latentvans/scripts
CK=/scratch/users/anirban/tamaghnam/latentvans/checkpoints/vdm_pilot
R=/scratch/users/anirban/tamaghnam/latentvans/results/vans_eval
log() { echo "$(date +%H:%M) $*"; }
sub() { bash tmux_job.sh "$@" | tail -1 | grep -o "job [0-9]*" | cut -d" " -f2; }
wait_slots() { while [ $(squeue -u $USER -h | wc -l) -ge 4 ]; do sleep 120; done; }
BUILD=${BUILD_JOB:-66150}
J_VANS=$(sub vans_eval_vans ARM=vans sbatch --partition=ada --dependency=afterok:$BUILD vans_eval_generate.sh); log "vans arm $J_VANS"
J_OR=$(sub vans_eval_vans_oracle ARM=vans_oracle sbatch --partition=ada --dependency=afterok:$BUILD vans_eval_generate.sh); log "vans_oracle arm $J_OR"
wait_slots
J_N=$(sub vans_eval_pilot_null ARM=pilot NAME=pilot_null CKPT=$CK/null_vlm_quad_abs/step_3000 CAPTIONS=null sbatch --partition=ada --dependency=afterok:$BUILD vans_eval_generate.sh); log "pilot_null $J_N"
wait_slots
J_O=$(sub vans_eval_pilot_oracle ARM=pilot NAME=pilot_oracle CKPT=$CK/caption_vlm_quad_abs_text-oracle/step_3000 CAPTIONS=oracle sbatch --partition=ada --dependency=afterok:$BUILD vans_eval_generate.sh); log "pilot_oracle $J_O"
wait_slots
J_C=$(sub vans_eval_pilot_caption ARM=pilot NAME=pilot_caption CKPT=$CK/caption_vlm_quad_abs/step_3000 "CAPTIONS=$R/vans/captions_shard*.jsonl" sbatch --partition=ada --dependency=afterok:$J_VANS vans_eval_generate.sh); log "pilot_caption $J_C"
log "all eval arms submitted"
