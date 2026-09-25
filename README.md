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
  panda70m_frames/              pre-extracted Stage 0 frames (+ extract_report.jsonl, extract_summary.json)
  broken_and_missing_clips.txt  input list for RECUT_FROM_LIST
checkpoints/
  Qwen2.5-VL-3B-Instruct/       base VLM
  stage0_futurel150k_run1/      Stage 0 LoRA run on Future-L1-50K (step_600 ... step_2400, metrics.jsonl)
logs/                           Slurm <job-name>_<jobid>.out/.err
code/VANS                       upstream submodule (see below); code/TwiFF is an empty clone
```

Not tracked (see `.gitignore`): `data/` (~61 GB), `checkpoints/` (~17 GB), `logs/`, `envs/`, wandb run dirs, `cookies.txt`.

## Upstream code

`code/VANS` is an unmodified clone of https://github.com/KlingAIResearch/VANS at commit `4a931a8`:

```bash
git clone https://github.com/KlingAIResearch/VANS.git code/VANS && git -C code/VANS checkout 4a931a8
```
