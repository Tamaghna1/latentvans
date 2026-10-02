#!/bin/bash
# Submit a Slurm job and give it its own tmux session that follows its log.
#
#   bash tmux_job.sh <session_name> [VAR=value ...] <script.sh> [sbatch options before the script]
#   e.g. bash tmux_job.sh s0v2_qdelta TARGET_DELTA=1 sbatch --job-name=s0v2_qdelta train_stage0_latent_grounding.sh
#
# Or attach a session to a job that is already running:
#
#   bash tmux_job.sh --attach <session_name> <jobid>
#
# Session layout (name = <session_name>_<jobid>):
#   window "log"    tail -F of the job's stdout (scrollback kept at 200k lines)
#   window "err"    tail -F of the job's stderr
#   window "status" squeue/sacct for the job, refreshed every 60 s
# The session stays after the job ends, so the full output can still be read with
# `tmux a -t <session>`; the log files under logs/ remain the permanent record.
set -euo pipefail

make_session() {
    local name="$1" jobid="$2"
    local session="${name}_${jobid}"
    local out err
    out=$(scontrol show job "$jobid" 2>/dev/null | sed -n 's/^ *StdOut=//p')
    err=$(scontrol show job "$jobid" 2>/dev/null | sed -n 's/^ *StdErr=//p')
    if [[ -z "$out" ]]; then
        # Finished jobs drop out of scontrol; fall back to the logs/<name>_<jobid> convention.
        out=$(ls /scratch/users/anirban/tamaghnam/latentvans/logs/*_"$jobid".out 2>/dev/null | head -1)
        err="${out%.out}.err"
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session $session already exists"
        return
    fi
    tmux new-session -d -s "$session" -n log
    tmux set-option -t "$session" history-limit 200000
    tmux send-keys -t "$session:log" "tail -n +1 -F '$out'" Enter
    tmux new-window -t "$session" -n err
    tmux send-keys -t "$session:err" "tail -n +1 -F '$err'" Enter
    tmux new-window -t "$session" -n status
    tmux send-keys -t "$session:status" \
        "while true; do clear; date; squeue -j $jobid 2>/dev/null; sacct -j $jobid -X --format=JobID,JobName%24,State,Elapsed,NodeList; sleep 60; done" Enter
    tmux select-window -t "$session:log"
    echo "job $jobid -> tmux session $session (attach: tmux a -t $session)"
}

if [[ "${1:-}" == "--attach" ]]; then
    make_session "$2" "$3"
    exit 0
fi

name="$1"; shift
env_vars=()
while [[ $# -gt 0 && "$1" == *=* ]]; do env_vars+=("$1"); shift; done
[[ "${1:-}" == "sbatch" ]] && shift
jobid=$(env "${env_vars[@]}" sbatch --parsable "$@")
jobid="${jobid%%;*}"
echo "submitted $jobid"
make_session "$name" "$jobid"
