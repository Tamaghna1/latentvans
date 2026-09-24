#!/bin/bash
#SBATCH --job-name=screen_panda70m_clip_quality
#SBATCH --partition=ada
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/screen_panda70m_clip_quality_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/screen_panda70m_clip_quality_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail

# Runs screen_panda70m_clip_quality.py -- see that file's module docstring
# for the full story (why manual spot-checking doesn't scale to 1879 clips,
# what "VLM as a zero-shot judge" means here, and the recommended workflow
# after this job finishes). Short version: this is inference-only (no
# training, no gradient, no LoRA) over the already-downloaded clips, using
# the same Qwen2.5-VL-3B-Instruct checkpoint train_stage0 uses.
#
# --mem=32G: a guess, not yet measured -- this is much lighter than Stage 0
# training (one frozen 3B VLM doing generate() on a few images, no
# optimizer/gradient state at all) but hasn't been confirmed on a real run
# yet. Check `sacct -j <id> --format=MaxRSS` after this runs once and lower
# it if there's a lot of headroom, same as every other --mem value in this
# project.
#
# --time=08:00:00: a rough budget for ~1879 clips at a few seconds each of
# frame extraction + VLM generate(). Not measured yet either. This is safe
# to just resubmit if it times out partway -- screen_panda70m_clip_quality.py
# skips clips already recorded in its report file (--report_path), same
# "safe to resubmit" pattern as download_panda70m_clips.py.
#
# Deliberately does NOT pip install/upgrade torch, torchvision, torchaudio,
# or anything else here -- transformers + torch should already be present
# in this env from earlier setup (train_stage0_latent_grounding.sh already
# depends on them), and this script needs no new packages beyond what's
# already installed (no decord, no peft -- this is read-only inference, not
# training). Given this project's whole ncclCommResume saga (see
# claude/latentvans-handoff.md), the safest thing an inference-only script
# like this one can do is touch the pinned torch stack NOT AT ALL rather
# than add one more "harmless-looking" install line -- just a preflight
# import check below to fail fast and clearly if something's missing,
# instead of guessing at a fix by installing something.

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"

QWEN_DIR="$SCRATCH/checkpoints/Qwen2.5-VL-3B-Instruct"
METADATA_DIR="$SCRATCH/data/twiff_metadata_2000"
CLIPS_DIR="$SCRATCH/data/panda70m_clips"

echo "===== Screen Panda-70M Clip Quality (VLM zero-shot judge) ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi || true

mkdir -p "$SCRATCH/logs"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

# Same defensive fix as the other GPU job templates in this project -- see
# vlm_vdm_bridge_smoke_test.sh's own comment for the full ncclCommResume
# story. Prepend the env's own site-packages nvidia/torch lib dirs so those
# always win over anything else on LD_LIBRARY_PATH.
SITE_PACKAGES="$ENV/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/torch/lib:${LD_LIBRARY_PATH:-}"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffmpeg/ffprobe not found on PATH -- installing via conda-forge..." >&2
    conda install -y -c conda-forge ffmpeg
fi

# Preflight: fail loudly and helpfully right here if torch/transformers can't be
# imported, rather than partway through loading a 3B model. Installs nothing --
# see the note above for why that's deliberate.
python - <<'PYEOF'
import sys
try:
    import torch
    import transformers
except ImportError as e:
    print(f"FATAL: import failed: {e}", file=sys.stderr)
    print("This script deliberately does not pip install anything -- torch/transformers should", file=sys.stderr)
    print("already be present from earlier setup (train_stage0_latent_grounding.sh's own deps).", file=sys.stderr)
    print("If they're genuinely missing, install them the same pinned way vlm_vdm_bridge_smoke_test.sh", file=sys.stderr)
    print("does (torch==2.5.1+cu121, transformers==5.16.1) rather than a bare pip install here.", file=sys.stderr)
    sys.exit(1)
print(f"torch import OK: {torch.__version__} (cuda build {torch.version.cuda})")
print(f"transformers import OK: {transformers.__version__}")
PYEOF

if [[ ! -d "$CLIPS_DIR" ]] || [[ -z "$(ls -A "$CLIPS_DIR"/*.mp4 2>/dev/null)" ]]; then
    echo "No .mp4 files found under $CLIPS_DIR -- run download_panda70m_clips.sh first." >&2
    exit 1
fi

# Expects screen_panda70m_clip_quality.py copied to $SCRATCH/scripts alongside
# the other templates (same convention as every other .sh/.py pair here).
python "$SCRATCH/scripts/screen_panda70m_clip_quality.py" \
    --qwen_dir "$QWEN_DIR" \
    --metadata_dir "$METADATA_DIR" \
    --clips_dir "$CLIPS_DIR" \
    --num_frames 3 \
    --dtype bf16 

echo "===== Screen Panda-70M clip quality completed — see $CLIPS_DIR/quality_screen_flagged.json ====="
echo "NEXT: manually eyeball a good chunk of quality_screen_flagged.json (frames are already saved"
echo "under $CLIPS_DIR/_quality_screen_frames/), AND a small random sample of MATCH-verdict clips too"
echo "-- see screen_panda70m_clip_quality.py's module docstring for why both checks matter."
echo "End: $(date)"
