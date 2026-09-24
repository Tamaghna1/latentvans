#!/bin/bash
#SBATCH --job-name=vlm_vdm_bridge_smoke_test
#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/vlm_vdm_bridge_smoke_test_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/vlm_vdm_bridge_smoke_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

# NOTE on --partition: moved off `a100` to `ada` on 2026-09-06. `a100`
# (single node, cn6) rejected this job outright with PD reason
# `(MaxMemPerLimit)` at BOTH --mem=128G and --mem=64G -- a hard configured
# ceiling below 64G, not the old "silently capped to 56G" behavior seen
# before. Rather than guess a third number on the smallest/most contested
# partition here, just run on `ada` (cn7-9, multiple nodes) instead -- it's
# already the partition recommended for faster queueing anyway. If you
# specifically need `a100`, get its real ceiling first with
# `scontrol show partition a100` (look for MaxMemPerNode/MaxMemPerCPU in the
# output) rather than trial-and-error.
#
# NOTE on --mem: originally bumped 64G -> 128G after a real run OOM-killed at
# the cgroup (host RAM) level, right at optimizer.step() -- but that OOM
# happened BEFORE --freeze_vlm was made the default invocation below.
# Freezing the VLM removes its AdamW optimizer state and gradients entirely
# (the actual source of that OOM), so 64G should comfortably cover this
# script's real footprint (two frozen/mostly-frozen models loaded with
# low_cpu_mem_usage=True, one optimizer step over just Proj + the Wan
# transformer). Not yet verified against `ada`'s own MaxMemPerNode -- check
# `sacct -j <id> --format=ReqMem` after submitting, same as always.

# Runs vlm_vdm_bridge_smoke_test.py -- see that file's module docstring for
# what it checks and why the two Wan checkpoint directories differ.
#
# Adjust #SBATCH --partition above to whichever GPU partition you want
# (a100 / ada / h200). `ada` is the default here -- `a100` (cn6, single node)
# turned out to have a memory ceiling below 64G (see the --partition note
# above) on top of being the most queue-contested partition on this cluster,
# so it's no longer assumed to be the safe default for even a small
# batch-size-1 smoke test.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
CHECKPOINT_DIR="$SCRATCH/checkpoints"

QWEN_DIR="$CHECKPOINT_DIR/Qwen2.5-VL-3B-Instruct"
WAN_DIFFUSERS_DIR="$CHECKPOINT_DIR/Wan2.1-T2V-1.3B-Diffusers"

echo "===== VLM -> VDM Bridge Smoke Test ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true   # full header (includes Driver Version / CUDA Version) -- needed to catch
                     # torch-vs-driver CUDA mismatches; --query-gpu alone suppresses this header

mkdir -p "$SCRATCH/logs"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

# ACTUAL ROOT CAUSE of the recurring "undefined symbol: ncclCommResume"
# ImportError, found 2026-09-07 (the LD_LIBRARY_PATH prepend below was a
# reasonable first guess but did NOT fix it, because it wasn't the problem):
# this script used to end its dependency install with
#   pip install --upgrade 'transformers>=4.49.0' 'diffusers>=0.32.0' accelerate pillow numpy torchvision
# -- note torchvision in that list, installed with a bare `--upgrade` and NO
# --index-url. Every time this script ran, that line silently pulled the
# LATEST torchvision from plain PyPI, which drags in whatever (newer,
# unpinned, possibly cu13x-built) torch version IT requires as a
# dependency -- overwriting the carefully pinned `torch==2.5.1+cu121` below
# on every single re-submission, including ones after the pin had been
# freshly confirmed working. A newer accidental torch build referencing a
# brand-new NCCL API (ncclCommResume) than the also-reinstalled
# nvidia-nccl-cu12 provides is exactly the symbol-mismatch error seen.
# FIX: pin the whole torch stack HERE (so the script is self-contained and
# reproducible instead of depending on whatever was last installed by hand),
# and never again upgrade transformers/diffusers/accelerate/torchvision
# together in one bare `--upgrade` call -- pin transformers/diffusers to
# known-working versions too, and leave torch/torchvision/torchaudio
# completely alone in every other pip command in this script.
python -m pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Kept as defense-in-depth (doesn't hurt, wasn't the actual bug this time):
# if something on this node's LD_LIBRARY_PATH ever does shadow this env's
# own torch-bundled CUDA libs, this makes the env's own copies win anyway.
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

# Preflight: fail loudly and helpfully right here, before spending time on
# the rest of the job, if torch still can't be imported.
python - <<'PYEOF'
import sys
try:
    import torch
except ImportError as e:
    print(f"FATAL: `import torch` failed: {e}", file=sys.stderr)
    print("If this is the ncclCommResume symbol error again: check that nothing later in this", file=sys.stderr)
    print("script (or a stray manual `pip install --upgrade` on the login node) re-touched", file=sys.stderr)
    print("torch/torchvision/torchaudio without --index-url .../cu121 -- that's the confirmed root", file=sys.stderr)
    print("cause, not an LD_LIBRARY_PATH shadowing issue. See claude/latentvans-handoff.md.", file=sys.stderr)
    sys.exit(1)
print(f"torch import OK: {torch.__version__} (cuda build {torch.version.cuda})")
PYEOF

if [[ ! -d "$WAN_DIFFUSERS_DIR" ]]; then
    echo "Missing $WAN_DIFFUSERS_DIR -- run download_wan_diffusers_checkpoint.sh first." >&2
    exit 1
fi

# Pinned, NOT --upgrade, and deliberately excludes torch/torchvision/torchaudio
# -- see the note above. transformers==5.16.1 / diffusers==0.40.0 are the
# exact versions already confirmed working against torch 2.5.1 on this
# project (see claude/latentvans-handoff.md's dependency-pinning-gap note).
python -m pip install --no-cache-dir \
    transformers==5.16.1 diffusers==0.40.0 'accelerate>=0.34,<1.0' pillow numpy

# Expects vlm_vdm_bridge_smoke_test.py copied to $SCRATCH alongside the
# other templates (same convention as data_staging.sh's siblings).
# --freeze_vlm: a prior run confirmed the full VLM->Proj->Wan forward+backward
# graph is wired correctly (grad norm printed successfully) but then got
# cgroup-OOM-killed at optimizer.step() with the VLM trainable -- AdamW state
# for ~3B extra params was the likely tipping point. Freezing the VLM here
# also matches the real Stage 1/2 recipe (VLM updated via GRPO, not direct
# backprop). Drop --freeze_vlm only once this passes comfortably within the
# --mem budget above and you specifically want to smoke-test a Stage-0-style
# full joint backward.
python "$SCRATCH/scripts/vlm_vdm_bridge_smoke_test.py" \
    --qwen_dir "$QWEN_DIR" \
    --wan_diffusers_dir "$WAN_DIFFUSERS_DIR" \
    --dtype bf16 \
    --freeze_vlm

echo "===== VLM -> VDM bridge smoke test completed successfully ====="
echo "End: $(date)"
