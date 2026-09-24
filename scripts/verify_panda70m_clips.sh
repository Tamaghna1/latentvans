#!/bin/bash
set -euo pipefail

# Spot-checks the clips download_panda70m_clips.sh already fetched -- see
# verify_panda70m_clips.py's module docstring for what it does and why.
# Deliberately NOT an sbatch job: it only reads files already on disk and
# runs ffprobe/ffmpeg thumbnail extraction over a handful of short clips --
# a few seconds of CPU work total, fine to run directly on the login node
# (same pattern as setup_env.sh: `bash verify_panda70m_clips.sh`).

SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"

METADATA_DIR="$SCRATCH/data/twiff_metadata_2000"
CLIPS_DIR="$SCRATCH/data/panda70m_clips"

echo "===== Verify Panda-70M Clips (spot-check) ====="
echo "Start: $(date)"

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "ffprobe not found on PATH -- it ships with ffmpeg (already installed by "
    echo "download_panda70m_clips.sh). Run that first, or 'conda install -y -c conda-forge ffmpeg'." >&2
    exit 1
fi

python "$SCRATCH/scripts/verify_panda70m_clips.py" \
    --metadata_dir "$METADATA_DIR" \
    --clips_dir "$CLIPS_DIR" \
    --num_thumbnails 3

echo "===== Verify Panda-70M clips completed — see $CLIPS_DIR/spot_check_report.json ====="
echo "End: $(date)"
