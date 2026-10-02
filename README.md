# latentvans

Training and data scripts for the latentvans project (stage-0 latent grounding on Panda-70M clips).

## Layout

Paths below are hardcoded in the Slurm scripts (`$SCRATCH/...`), so don't move them.

```
scripts/                        active Slurm .sh wrappers + Python (staging, download, screening, frame extraction, stage-0 train/eval)
  legacy/                       superseded phaseN_* scripts and download_panda70m_clips_v2.py (reference only, not used)
  wandb/                        local W&B run dirs (written by training)
data/
  twiff_metadata_2000/          TwiFF-2.7M first-2000-row train slice
  twiff_metadata_val/           TwiFF-2.7M official validation split (1,451 rows), held-out set
  future_l1_50k_metadata_full/  Future-L1-50K metadata, current Stage 0 training set
  panda70m_clips/               downloaded train clips (+ download_summary.json, _raw_downloads/ yt-dlp staging)
  panda70m_clips_val/           downloaded validation clips (+ quality screen reports)
  panda70m_frames/              OLD frame cache, WRONG frames (pre-2026-10-02 index bug, see below); superseded by _v2
  panda70m_frames_v2/           pre-extracted Stage 0 train frames, correct TwiFF indexing (+ extract_report.jsonl, extract_summary.json)
  panda70m_frames_val_v2/       same, validation split
  stage0_targets_v2/            target analysis + z-score stats from analyze_stage0_targets.py
  broken_and_missing_clips.txt  input list for RECUT_FROM_LIST
checkpoints/
  Qwen2.5-VL-3B-Instruct/       base VLM
  stage0_futurel150k_run1/      Stage 0 LoRA run on Future-L1-50K (step_600 ... step_2400, metrics.jsonl)
logs/                           Slurm <job-name>_<jobid>.out/.err
code/VANS                       upstream submodule (see below); code/TwiFF is an empty clone
```

Not tracked (see `.gitignore`): `data/` (~61 GB), `checkpoints/` (~17 GB), `logs/`, `envs/`, wandb run dirs, `cookies.txt`.

## TwiFF frame indexing (fixed 2026-10-02)

`question_images_index` / `reasoning_images_index` in TwiFF-2.7M and Future-L1-50K are positions
1..8 among eight frames TwiFF sampled from each cut clip, at fractions 0, 1/12, 3/12, 5/12, 7/12,
9/12, 11/12 and 1 of the clip. They are not raw frame numbers. Until 2026-10-02 every script read
them as raw frame numbers, so Stage 0's context and "future" frames both came from the first
~0.3 s of each clip. That alone explains the earlier finding that the pooled future target was
96% similar to the context.

- `scripts/twiff_frames.py` has the mapping and an ffmpeg-based reader. It decodes clips that
  decord fails on (0 failures vs ~40%), and counts displayed frames rather than packets, since
  stream-copied cuts carry hidden pre-roll packets.
- `scripts/verify_twiff_frame_indexing.py` checks it against the JPEGs embedded in TwiFF's
  validation parquet: 92% of frames match, versus 33% for the old reading, which only gets index 1 right.
  The rest are a frame or two off, because our cuts differ slightly in length from TwiFF's.

## Stage 0 v2 workflow

```bash
cd $SCRATCH/scripts
sbatch extract_stage0_frames.sh                                   # -> data/panda70m_frames_v2
METADATA_DIR=$SCRATCH/data/twiff_metadata_val VIDEO_ROOT=$SCRATCH/data/panda70m_clips_val \
  OUTPUT_DIR=$SCRATCH/data/panda70m_frames_val_v2 sbatch extract_stage0_frames.sh
sbatch analyze_stage0_targets.sh                                  # target variants + z-score stats
sbatch train_stage0_latent_grounding.sh                           # v2 config, see the .sh header
CHECKPOINT=$SCRATCH/checkpoints/<run>/step_<N> sbatch diagnose_stage0_grounding.sh
```

`stage0_targets.py` defines the target options (`--target_layout pooled|quadrants`, `--target_delta`,
`--target_stats`). Training also takes `--latent_head linear`. CE no longer supervises the
`<|latent_pad|>` positions. Each checkpoint saves `stage0_target_config.json`, which the eval and
diagnose scripts read.

## Upstream code

`code/VANS` is an unmodified clone of https://github.com/KlingAIResearch/VANS at commit `4a931a8`:

```bash
git clone https://github.com/KlingAIResearch/VANS.git code/VANS && git -C code/VANS checkout 4a931a8
```
