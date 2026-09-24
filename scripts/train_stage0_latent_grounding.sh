#!/bin/bash
#SBATCH --job-name=stage0_sft
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/stage0_sft_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/stage0_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
set -euo pipefail

# Stage 0 latent-grounding SFT (see train_stage0_latent_grounding.py's module
# docstring for what this actually does and why). Unlike the VLM->VDM bridge
# smoke test (vlm_vdm_bridge_smoke_test.sh), the VLM is trainable here BY
# DESIGN -- this is real, possibly multi-hour training, not a shape-check.
#
# LoRA by default (see train_stage0_latent_grounding.py's "LoRA fine-tuning"
# docstring section): the current Panda-70M clip set (~1881 clips, post the
# 2026-09-09 stale-file-bug fix -- see claude/latentvans-handoff.md item 5)
# is still small relative to the research plan's VANS-Data-100K/TwiFF-2.7M
# assumptions, so LoRA stays the safer default over full-parameter
# fine-tuning. --mem=90G below is a deliberate margin under ada's confirmed
# hard cap (2026-09-09, via `scontrol show partition ada`): MaxMemPerNode=98304
# (96 GiB) per job, DefMemPerNode=65536 (64 GiB) if --mem is left unset. The
# original 128G guess this replaced would have exceeded that cap and never
# started (same failure mode a100 hit at even lower requests -- see
# claude/latentvans-handoff.md SLURM gotchas). This has still not been
# re-measured under a real LoRA training run -- check `sacct -j <id>
# --format=MaxRSS` after this first real run and lower it toward the actual
# number rather than leaving it at this ceiling-driven guess.
#
# QUALITY-SCREEN EXCLUSION (2026-09-09): $EXCLUDE_FLAGGED_JSON below points at
# screen_panda70m_clip_quality.py's quality_screen_flagged.json -- every clip
# that automated screen called MISMATCH/PARSE_ERROR/NO_METADATA. If that file
# exists, this run passes it via --exclude_flagged_json so those flagged rows
# are never drawn from at all (see train_stage0_latent_grounding.py's module
# docstring, "Quality-screen filtering"). Per explicit instruction, this run
# proceeds without first manually re-verifying the remaining MATCH clips --
# note that in your run log so it's clear this was a deliberate choice, not
# an oversight (the screen is a heuristic with at least one known false
# negative -- see claude/latentvans-handoff.md item 5).
#
# --partition=short (2026-09-24, changed from ada): confirmed the actual
# working partition for this job -- see the GPU-partition-hunt entry in
# claude/latentvans-training-decisions-log.md. Both `long` (cn1, job 61173)
# and `h200` (cn10) allocated a GPU via SLURM but then failed at the
# nvidia-smi/NVML level (node-specific hardware/driver faults); `short`
# (cn2/cn5, A5000 GPUs) is the one partition confirmed to actually train
# end-to-end (job 61184 ran real steps on it). `short`'s own wall-clock
# ceiling is 24h (see claude/kiac_slurm_templates -- cluster-and-workflow
# notes), matching --time=24:00:00 below exactly -- MAX_STEPS is sized
# against that ceiling (see "MAX_STEPS / SAVE_EVERY RECALIBRATED" below).
# If `short` is queued/busy, check `squeue` and reconsider, but re-verify
# nvidia-smi/torch.cuda.is_available() actually works on whatever partition
# you fall back to -- `long`/`h200` looked fine at allocation time and
# weren't.
#
# --time=24:00:00: `short`'s own hard ceiling, not a conservative choice --
# see above. Use --resume_from across multiple job submissions to keep
# training past what one job's 24h can cover, rather than expecting a
# longer --time to work here.
#
# WEIGHTS & BIASES (2026-09-09, OPT-IN, ENABLED 2026-09-24): this project's
# handoff doc has said "No Weights & Biases" since early on -- that was
# about not silently depending on an external service nothing else in this
# project used. This is a deliberate, explicit exception for this one
# script, at the user's request, so loss curves can be watched live instead
# of only via metrics.jsonl. $WANDB_PROJECT below is now set (see
# "ATTACHED" note by the variable itself) -- it stays functionally off if
# `wandb login` was never run, since W&B init failure is caught and logged
# as a WARNING, not a FATAL (see the .py's own docstring): training
# proceeds either way, metrics.jsonl is unaffected. BEFORE this run, on the
# KIAC login node, run `wandb login` ONCE (pastes an API key interactively,
# writes it to ~/.netrc, and compute nodes pick that up too since they
# share $HOME) -- or export WANDB_API_KEY in this job's environment
# instead. Either way, no API key ever goes on the command line here on
# purpose: CLI args land in job logs and `ps` output, a wandb credential
# should not. If the compute node has no internet egress, W&B init fails
# safely and training proceeds without it.
#
# HELD-OUT VALIDATION (added 2026-09-12): $VAL_METADATA_DIR/$VAL_VIDEO_ROOT
# below point at the official TwiFF-2.7M `validation` split (1,451 rows,
# disjoint from the 2,000-row `train` slice everything else in this project
# has used so far) -- see data_staging.sh's validation-split staging block
# and download_panda70m_clips.sh/screen_panda70m_clip_quality.sh's own
# env-var overrides for how to actually populate $VAL_VIDEO_ROOT with real
# clips (this script does NOT fetch them itself). If $VAL_METADATA_DIR
# exists when this job runs, --val_metadata_dir/--val_video_root are passed
# through automatically and periodic held-out eval turns on -- see
# train_stage0_latent_grounding.py's module docstring, "Held-out
# validation", for why this matters: job 57910's train loss collapsed to
# near-zero by ~step 150-200 and stayed flat for the rest of a 2000-step
# run, which train loss alone cannot distinguish from real learning. This
# validation split stays TwiFF's official one, NOT a Future-L1-50K slice --
# check_train_val_overlap.sh confirmed (2026-09-24) zero/negligible overlap
# between it and Future-L1-50K's train rows, so it remains a genuinely
# held-out set for $METADATA_DIR below even though that's now Future-L1-50K,
# not the original TwiFF train slice this validation split was first
# checked against. If $VAL_METADATA_DIR does NOT exist yet, this run
# proceeds WITHOUT held-out eval (a warning is printed, both here and from
# the python script) rather than failing.
#
# PRE-EXTRACTED FRAME CACHE (added 2026-09-24): $FRAMES_ROOT/$VAL_FRAMES_ROOT
# below point at extract_stage0_frames.py's output (a directory of just the
# frames this run actually reads, one subdirectory per clip -- see that
# script's own module docstring and train_stage0_latent_grounding.py's
# "Pre-extracted frame cache" section for the full rationale: it avoids
# re-paying decord's ~30-41% decode-failure rate on every training step,
# and shrinks the working set versus keeping full raw clips around). Each
# is auto-detected the same way $VAL_METADATA_DIR already is: if the
# directory exists when this job runs, --frames_root/--val_frames_root are
# passed through and that stream reads pre-extracted JPEGs instead of
# live-decoding raw video; if it doesn't exist yet, that stream falls back
# to the original live-decode behavior with a warning, not a failure.
# Blank (default) = both unset = fully unchanged prior behavior (live
# decode from $VIDEO_ROOT/$VAL_VIDEO_ROOT).
#
# IMPORTANT: do NOT delete or move $VIDEO_ROOT's raw clips off this cluster
# just because $FRAMES_ROOT is set -- $VIDEO_ROOT stays the live-decode
# fallback for any clip extract_stage0_frames.py hasn't (yet) covered for
# whatever --metadata_dir this run uses, and is needed again if the
# metadata slice is ever restaged larger (e.g. resuming the Future-L1-50K
# download past its current 42,214/50,000 clips) or if extraction is ever
# re-run with different settings. Confirm extract_summary.json's coverage
# for the metadata slice actually in use, and that both train AND val
# frame caches are populated, before treating the raw clips as disposable.
#
# METADATA_DIR / OUTPUT_DIR CORRECTED (2026-09-24): this file was found
# still pointing at the ORIGINAL small TwiFF pilot (METADATA_DIR=
# twiff_metadata_2000, OUTPUT_DIR=stage0_run1) when this correction was
# made, even though claude/latentvans-training-decisions-log.md's
# 2026-09-23 entry records METADATA_DIR having been switched to
# future_l1_50k_metadata_full -- that switch was apparently never actually
# persisted to this file (or was reverted at some point without being
# caught). Submitting this script as it was would have (a) silently
# trained on the tiny ~2,000-row TwiFF pilot instead of the ~29,479+
# currently-clean Future-L1-50K clips this whole session's work has been
# building toward, and (b) written a fresh run's checkpoints into
# stage0_run1/ -- the ORIGINAL job 57910 baseline directory that was
# explicitly kept untouched for comparison (see the 2026-09-24
# "MAX_STEPS/SAVE_EVERY recalibrated" decision log entry). Both are fixed
# below: METADATA_DIR -> future_l1_50k_metadata_full, OUTPUT_DIR ->
# stage0_futurel150k_run1 (a genuinely fresh directory -- RESUME_FROM stays
# blank). Double-check these two values against the decision log any time
# this file is touched again; this is exactly the kind of silent drift that
# wastes a multi-hour GPU run on the wrong data without any error.
#
# MAX_STEPS / SAVE_EVERY RECALIBRATED (2026-09-24): the old default
# (--max_steps 2000) was sized against the ORIGINAL ~1800-1900 clip TwiFF
# pilot -- 2000 steps * grad_accum 8 = 16,000 example draws, ~8-9 epochs
# over that small pool (see train_stage0_latent_grounding.py's "LoRA
# fine-tuning" docstring section for the original reasoning: enough passes
# to learn something, few enough to bound memorization risk on a small,
# repeated pool). That reasoning was never revisited after switching to
# Future-L1-50K (~42,214 clips, ~23x more data) -- at the same 2000 steps,
# 16,000 draws is under 0.4 epochs over the new pool, and job 61184's own
# val_loss_total was STILL DECLINING at every checkpoint through step 1000
# (1.5415->1.4652, no plateau) when it crashed -- the opposite signature
# from what 2000 was originally guarding against. MAX_STEPS below is now
# 6000 (~1.1 epochs over the current pool at grad_accum 8), derived from
# job 61184's own observed throughput on `short` (~1000 steps in ~3h49m,
# ~262 steps/hour) to fit inside a single --time=24:00:00 job with margin
# for eval/checkpoint overhead -- NOT a benchmarked constant, since
# throughput varies by which node this lands on (see the GPU-partition-hunt
# entry in claude/latentvans-training-decisions-log.md) and by whether
# --frames_root is active (removes decord's per-step decode cost entirely
# once the extracted-frame cache covers this run's clips). As of this
# correction, the currently-clean Future-L1-50K frame cache is ~29,479
# clips (an initial extraction pass; a broken/missing-clip recut is
# separately in progress as of this entry, per the decision log -- more
# clips may become available mid-run without needing to restart, since a
# resubmit/resume just re-reads whatever's on disk at that time), not the
# full ~42,214-50,000 -- MAX_STEPS was sized against the larger pool
# assumption and is even more conservative (fewer effective epochs) against
# this actual current count; revisit upward once more clips are confirmed
# clean if that matters to you. Check `sacct -j <id> --format=Elapsed` after
# this run and adjust up or down; going past a single job's wall-clock
# budget means using --resume_from across multiple submissions, not raising
# --time past `short`'s 24h ceiling.
# SAVE_EVERY raised from 200 to 600 in proportion (keeps ~10 checkpoints
# total over the run, same density as before, rather than tripling
# checkpoint count/disk usage for the same wall-clock budget -- this
# project has already hit disk quota exhaustion twice, most recently
# fatally mid-run, so checkpoint count is not free to let float upward).

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"

QWEN_DIR="$SCRATCH/checkpoints/Qwen2.5-VL-3B-Instruct"
METADATA_DIR="$SCRATCH/data/future_l1_50k_metadata_full"          # CORRECTED 2026-09-24 -- was twiff_metadata_2000, see comment block above
VIDEO_ROOT="$SCRATCH/data/panda70m_clips"
EXCLUDE_FLAGGED_JSON="$VIDEO_ROOT/quality_screen_flagged.json"   # from screen_panda70m_clip_quality.py -- NOT yet run against Future-L1-50K (see decision log, 2026-09-23 entry, "quality screen explicitly skipped"); this file won't exist, so EXCLUDE_ARGS below will be empty and a WARNING will print -- that's expected, not a bug
OUTPUT_DIR="$SCRATCH/checkpoints/stage0_futurel150k_run1"          # CORRECTED 2026-09-24 -- was stage0_run1 (job 57910's baseline dir, must stay untouched), see comment block above
RESUME_FROM=""                                     # set to a prior step_NNNN dir to resume -- stays blank, this is a fresh run into a fresh OUTPUT_DIR
MAX_STEPS=6000                                      # see "MAX_STEPS / SAVE_EVERY RECALIBRATED" comment above
SAVE_EVERY=600                                       # keeps ~10 checkpoints over the run, same density as the old 2000/200
WANDB_PROJECT="latentvans-stage0"                  # ATTACHED 2026-09-24 at explicit user request -- requires `wandb login` once on the login node first (see comment block above); if that hasn't been done, this fails safely and training proceeds without it
WANDB_ENTITY=""                                    # optional -- your W&B username/team; blank = account default
WANDB_RUN_NAME=""                                  # optional -- blank = script default (stage0-<job id>)

# Held-out validation (see comment block above). Blank VAL_METADATA_DIR
# disables eval entirely -- leave these as-is once data_staging.sh's
# validation block and the two clip-fetching scripts have populated them.
VAL_METADATA_DIR="$SCRATCH/data/twiff_metadata_val"
VAL_VIDEO_ROOT="$SCRATCH/data/panda70m_clips_val"
VAL_EXCLUDE_FLAGGED_JSON="$VAL_VIDEO_ROOT/quality_screen_flagged.json"
EVAL_EVERY=""                                       # blank = defaults to --save_every in the python script
EVAL_MAX_EXAMPLES=200

# Pre-extracted frame cache (see comment block above). Blank = both streams
# live-decode raw video exactly as before this feature existed.
FRAMES_ROOT="${FRAMES_ROOT:-$SCRATCH/data/panda70m_frames}"
VAL_FRAMES_ROOT="${VAL_FRAMES_ROOT:-$SCRATCH/data/panda70m_frames_val}"

echo "===== Stage 0: Latent Grounding SFT ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true

mkdir -p "$SCRATCH/logs" "$OUTPUT_DIR"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

# Same defensive fix as vlm_vdm_bridge_smoke_test.sh -- see that script's
# comment for the full story: `import torch` can fail with "undefined
# symbol: ncclCommResume" if an older libnccl.so is found on LD_LIBRARY_PATH
# ahead of this env's own torch-bundled one. Prepend the env's own
# site-packages nvidia/torch lib dirs so those always win.
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -d "$VIDEO_ROOT" ]]; then
    echo "WARNING: $VIDEO_ROOT does not exist yet -- this run will exit fast with" >&2
    echo "NoUsableExamplesError. See the script's own note on Panda-70M clips." >&2
fi

EXCLUDE_ARGS=()
if [[ -f "$EXCLUDE_FLAGGED_JSON" ]]; then
    echo "Found $EXCLUDE_FLAGGED_JSON -- excluding its flagged clips from this run."
    EXCLUDE_ARGS=(--exclude_flagged_json "$EXCLUDE_FLAGGED_JSON")
else
    echo "WARNING: $EXCLUDE_FLAGGED_JSON not found -- proceeding WITHOUT quality-screen exclusion." >&2
    echo "(Run screen_panda70m_clip_quality.sh first if you want the flagged clips excluded.)" >&2
fi

VAL_ARGS=()
if [[ -d "$VAL_METADATA_DIR" ]]; then
    echo "Found $VAL_METADATA_DIR -- enabling held-out evaluation."
    VAL_ARGS=(--val_metadata_dir "$VAL_METADATA_DIR" --val_video_root "$VAL_VIDEO_ROOT" --eval_max_examples "$EVAL_MAX_EXAMPLES")
    if [[ -n "$EVAL_EVERY" ]]; then
        VAL_ARGS+=(--eval_every "$EVAL_EVERY")
    fi
    if [[ -f "$VAL_EXCLUDE_FLAGGED_JSON" ]]; then
        echo "Found $VAL_EXCLUDE_FLAGGED_JSON -- excluding its flagged clips from validation too."
        VAL_ARGS+=(--val_exclude_flagged_json "$VAL_EXCLUDE_FLAGGED_JSON")
    else
        echo "WARNING: $VAL_EXCLUDE_FLAGGED_JSON not found -- proceeding WITHOUT validation quality-screen exclusion." >&2
    fi
    if [[ ! -d "$VAL_VIDEO_ROOT" ]] || [[ -z "$(ls -A "$VAL_VIDEO_ROOT"/*.mp4 2>/dev/null)" ]]; then
        echo "WARNING: $VAL_VIDEO_ROOT has no .mp4 files yet -- eval will log 0 usable validation" >&2
        echo "examples per call until download_panda70m_clips.sh has been run against it." >&2
    fi
else
    echo "WARNING: $VAL_METADATA_DIR not found -- this run will have NO held-out evaluation." >&2
    echo "(Run data_staging.sh's validation-split staging block, then" >&2
    echo "  METADATA_DIR=$VAL_METADATA_DIR OUTPUT_DIR=$VAL_VIDEO_ROOT sbatch download_panda70m_clips.sh," >&2
    echo "  METADATA_DIR=$VAL_METADATA_DIR CLIPS_DIR=$VAL_VIDEO_ROOT sbatch screen_panda70m_clip_quality.sh," >&2
    echo "to enable it -- see train_stage0_latent_grounding.py's module docstring, 'Held-out validation'.)" >&2
fi

# Pre-extracted frame cache auto-detect (2026-09-24) -- same pattern as
# $VAL_METADATA_DIR above: present a covered directory and this run uses it
# automatically; absent, this run falls back to live-decode for that
# stream with a warning, not a failure. See comment block at the top of
# this file and train_stage0_latent_grounding.py's module docstring,
# "Pre-extracted frame cache".
FRAMES_ARGS=()
if [[ -n "$FRAMES_ROOT" ]] && [[ -d "$FRAMES_ROOT" ]]; then
    echo "Found $FRAMES_ROOT -- training stream will read pre-extracted frames instead of live-decoding $VIDEO_ROOT."
    FRAMES_ARGS+=(--frames_root "$FRAMES_ROOT")
else
    echo "WARNING: FRAMES_ROOT ($FRAMES_ROOT) not found -- training stream will live-decode raw video from \$VIDEO_ROOT." >&2
    echo "(Run extract_stage0_frames.sh first if you want this run to use the pre-extracted cache.)" >&2
fi
if [[ -d "$VAL_METADATA_DIR" ]]; then
    if [[ -n "$VAL_FRAMES_ROOT" ]] && [[ -d "$VAL_FRAMES_ROOT" ]]; then
        echo "Found $VAL_FRAMES_ROOT -- validation stream will read pre-extracted frames instead of live-decoding \$VAL_VIDEO_ROOT."
        FRAMES_ARGS+=(--val_frames_root "$VAL_FRAMES_ROOT")
    else
        echo "WARNING: VAL_FRAMES_ROOT ($VAL_FRAMES_ROOT) not found -- validation stream will live-decode raw video from \$VAL_VIDEO_ROOT." >&2
    fi
fi

# decord/peft/wandb all declare loose (not pinned) torch version requirements,
# so --upgrade here should not touch the already-pinned torch/torchvision/torchaudio
# stack -- pip only upgrades a dependency if what's installed doesn't satisfy the
# new package's constraint, and torch==2.5.1 comfortably does for all three. Still,
# given claude/latentvans-handoff.md's whole ncclCommResume saga, sanity-check
# `python -c "import torch; print(torch.__version__)"` after this if anything
# about the torch/CUDA env looks off following this run.
python -m pip install --upgrade decord peft wandb

ARGS=(
    --qwen_dir "$QWEN_DIR"
    --metadata_dir "$METADATA_DIR"
    --video_root "$VIDEO_ROOT"
    --output_dir "$OUTPUT_DIR"
    --dtype bf16
    --batch_size 1
    --grad_accum_steps 8
    --max_steps "$MAX_STEPS"
    --save_every "$SAVE_EVERY"
)
ARGS+=("${EXCLUDE_ARGS[@]}")
ARGS+=("${VAL_ARGS[@]}")
ARGS+=("${FRAMES_ARGS[@]}")
if [[ -n "$RESUME_FROM" ]]; then
    ARGS+=(--resume_from "$RESUME_FROM")
fi
if [[ -n "$WANDB_PROJECT" ]]; then
    echo "W&B logging requested: project=$WANDB_PROJECT"
    ARGS+=(--wandb_project "$WANDB_PROJECT")
    [[ -n "$WANDB_ENTITY" ]] && ARGS+=(--wandb_entity "$WANDB_ENTITY")
    [[ -n "$WANDB_RUN_NAME" ]] && ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
else
    echo "WANDB_PROJECT is empty -- W&B logging is off (metrics.jsonl still written as usual)."
fi

python "$SCRATCH/scripts/train_stage0_latent_grounding.py" "${ARGS[@]}"

echo "===== Stage 0 run finished (see log above for success/failure) ====="
echo "End: $(date)"
