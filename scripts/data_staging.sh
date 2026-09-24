#!/bin/bash
#SBATCH --job-name=data_staging
#SBATCH --partition=medium
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/data_staging_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/data_staging_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
SCRATCH="/scratch/users/anirban/tamaghnam/latentvans"
CONDA_ROOT="/scratch/users/anirban/tamaghnam/miniconda3"
ENV="/scratch/users/anirban/tamaghnam/envs/latentvans"
CODE_DIR="$SCRATCH/code"
CHECKPOINT_DIR="$SCRATCH/checkpoints"
DATA_DIR="$SCRATCH/data"
VANS_DIR="$CODE_DIR/VANS"
TWIFF_DIR="$CODE_DIR/TwiFF"
QWEN_DIR="$CHECKPOINT_DIR/Qwen2.5-VL-3B-Instruct"
WAN_DIR="$CHECKPOINT_DIR/Wan2.1-T2V-1.3B"

# TRAIN_DATA_SOURCE (added 2026-09-15, at explicit user request to use the
# Future-L1 paper's own pre-filtered subset instead of raw TwiFF): the
# Future-L1 paper (arXiv:2606.05769, "Imagine Before You Predict") did NOT
# train Stage 0 on all 2.71M TwiFF rows. It scored every TwiFF candidate by
# marginal visual-gain (p_v - p_t: how much does prediction improve when
# future frames are provided vs. text-only) using Qwen3-VL-8B-Instruct +
# Qwen3.5-397B-A17B as judge, 8 rollouts per condition, then kept the top
# 50,000 highest-gain examples as "Future-L1-50K". That scoring pipeline
# (~43M generations against a ~400B judge) is NOT reproducible on this
# cluster -- but the paper's authors published the RESULT of that filtering
# as a public HF dataset: https://huggingface.co/datasets/Eurayka/Future-L1-50K
# (confirmed via live fetch of the dataset page + repo, 2026-09-15). Using it
# gets you the same "train only on examples where future visual grounding
# actually helps" selection Future-L1 used, without running the filtering
# yourself.
#
# CITATION REQUIRED if TRAIN_DATA_SOURCE=future_l1_50k is used in the thesis
# (the HF dataset's license/card explicitly asks for it):
#   @article{jiang2026imagine,
#     title={Imagine Before You Predict: Interleaved Latent Visual Reasoning
#            for Video Event Prediction},
#     author={Jiang, Tianxiang and Wu, Linquan and Xia, Sheng and Li, Songze
#             and Yan, Ziang and Yang, Haoyu and Qiao, Yu and Wang, Yi},
#     journal={arXiv preprint arXiv:2606.05769},
#     year={2026}
#   }
#
# "twiff" (default) reproduces the exact previous behavior (raw TwiFF-2.7M
# `train` split, first TRAIN_SAMPLE_SIZE rows in streaming order) -- nothing
# changes if you don't set this.
#
# NOTE ON SCHEMA: Future-L1-50K's raw rows do NOT use TwiFF's own
# question/answer/question_images_index/reasoning_images_index column names
# -- they use a `conversations` field (list of {"from": "human"/"gpt",
# "value": ...} turns) plus `image`/`reasoning_image` int-list columns
# (confirmed via the HF dataset viewer, 2026-09-15). The conversion below
# normalizes future_l1_50k rows into TwiFF's own canonical column names at
# STAGING time, once, here -- not in train_stage0_latent_grounding.py /
# evaluate_stage0_latent_grounding.py. This is a deliberate choice: those two
# scripts already share (and have twice drifted out of sync on, see
# claude/latentvans-handoff.md) a single TwiFFStage0Stream implementation --
# adding a second schema branch there would mean touching and re-verifying
# both files for a schema difference that only exists at the source-dataset
# level. Doing the conversion once here means TwiFFStage0Stream needs ZERO
# changes and every downstream script keeps working exactly as already
# verified, regardless of which source produced the staged metadata_dir.
#
# The question-text extraction below strips leading "frame_N"/"<image>"
# markup lines from the human turn (TwiFF's own `question` column is plain
# question text with no such markup, and build_training_example() in
# train_stage0_latent_grounding.py already prepends its own {"type":"image"}
# entries from context_frames -- leaving the markup in would be redundant/
# confusing). This is a HEURISTIC based on a small viewer sample, not a
# verified-at-scale transform -- spot-check a handful of converted rows
# (see this job's own log) before trusting it across all 50,000.
#
# TRAIN_SAMPLE_SIZE still applies when TRAIN_DATA_SOURCE=future_l1_50k, but
# means something slightly different: since Future-L1-50K is already fully
# curated (not a raw 2.71M-row pool), taking a smaller TRAIN_SAMPLE_SIZE
# keeps the highest-`gain` rows (Future-L1's own ranking, descending) rather
# than an arbitrary prefix -- preserving the paper's selection principle
# instead of undoing it. Leave TRAIN_SAMPLE_SIZE unset to keep all 50,000.
#
# VALIDATION IS UNCHANGED regardless of TRAIN_DATA_SOURCE: Stage 0 held-out
# eval still uses TwiFF-2.7M's own official `validation` split (see below,
# unchanged from the 2026-09-12 addition) -- Future-L1-50K does not publish
# its own validation split, and evaluating against TwiFF's disjoint official
# split is arguably a cleaner generalization check anyway (data the curated
# training set was never drawn from at all, not just a held-out slice of it).
TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-twiff}"   # "twiff" (default) or "future_l1_50k"

if [[ "$TRAIN_DATA_SOURCE" == "future_l1_50k" ]]; then
    TRAIN_SAMPLE_SIZE="${TRAIN_SAMPLE_SIZE:-}"    # empty = keep all 50,000 rows
    SAMPLE_LABEL="${TRAIN_SAMPLE_SIZE:-full}"
    METADATA_DIR="${METADATA_DIR:-$DATA_DIR/future_l1_50k_metadata_${SAMPLE_LABEL}}"
else
    # TRAIN_SAMPLE_SIZE (added 2026-09-14, at explicit user request to increase the
    # number of DISTINCT training clips -- see claude/latentvans-handoff.md's
    # memorization-vs-generalization item): this used to be hardcoded to 2,000, an
    # arbitrary pilot-sized slice of TwiFF-2.7M's 2.71M-row `train` split. Now an
    # override -- default stays 2000 (unchanged behavior if you don't set it), but
    # set it higher to stage more rows and, downstream, download/screen/train
    # against a bigger, more diverse clip pool. Same idea as the existing
    # METADATA_DIR/OUTPUT_DIR overrides download_panda70m_clips.sh and
    # screen_panda70m_clip_quality.sh already support.
    #
    # IMPORTANT: `datasets`' streaming reads TwiFF-2.7M's `train` split in a fixed,
    # deterministic row order (not shuffled) -- so a bigger TRAIN_SAMPLE_SIZE's
    # first 2,000 rows are EXACTLY the same 2,000 rows the original default staged.
    # This matters practically: download_panda70m_clips.py's skip-if-exists resume
    # logic means you can point OUTPUT_DIR at the SAME clips directory you already
    # populated for the 2,000-row run, and it will skip every clip already on disk,
    # downloading only the newly-added rows -- no need for a separate output
    # directory or re-downloading anything already fetched, as long as
    # TRAIN_SAMPLE_SIZE only ever grows (never shrinks) between runs.
    #
    # METADATA_DIR is now named after the actual sample size (twiff_metadata_<N>)
    # rather than a fixed twiff_metadata_2000, so different-sized stagings don't
    # silently overwrite each other -- pass a matching METADATA_DIR to
    # download_panda70m_clips.sh / screen_panda70m_clip_quality.sh /
    # train_stage0_latent_grounding.sh afterward (see this job's own final log
    # line for the exact path to use).
    TRAIN_SAMPLE_SIZE="${TRAIN_SAMPLE_SIZE:-2000}"
    METADATA_DIR="${METADATA_DIR:-$DATA_DIR/twiff_metadata_${TRAIN_SAMPLE_SIZE}}"
fi
VAL_METADATA_DIR="$DATA_DIR/twiff_metadata_val"
echo "===== Data Staging: Checkpoints, Repos, TwiFF Metadata ====="
echo "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "TRAIN_DATA_SOURCE=$TRAIN_DATA_SOURCE"
echo "TRAIN_SAMPLE_SIZE=${TRAIN_SAMPLE_SIZE:-<unset -- all rows>}"
echo "METADATA_DIR=$METADATA_DIR"
mkdir -p "$CODE_DIR" "$CHECKPOINT_DIR" "$DATA_DIR" "$SCRATCH/logs"
cd "$SCRATCH"
if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found: $CONDA_ROOT/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
echo "Python: $(command -v python)"
python --version
python -m pip install --upgrade pip
python -m pip install --upgrade huggingface_hub datasets
if [[ ! -d "$VANS_DIR/.git" ]]; then
    if [[ -e "$VANS_DIR" ]]; then
        echo "Refusing to overwrite existing non-Git path: $VANS_DIR" >&2
        exit 1
    fi
    git clone https://github.com/AILab-CVC/VANS.git "$VANS_DIR"
else
    echo "VANS already exists; skipping clone."
fi
if [[ ! -d "$TWIFF_DIR/.git" ]]; then
    if [[ -e "$TWIFF_DIR" ]]; then
        echo "Refusing to overwrite existing non-Git path: $TWIFF_DIR" >&2
        exit 1
    fi
    git clone https://github.com/IntMeGroup/TwiFF.git "$TWIFF_DIR"
else
    echo "TwiFF already exists; skipping clone."
fi
# SKIP-IF-EXISTS (added 2026-09-15, at explicit user question "are you
# downloading the models again? i have them ready in my directory"): `hf
# download` is content-addressed/resumable so it won't re-pull bytes already
# present, but it still hits the network to verify every file against the
# hub on every single run -- pointless if $QWEN_DIR/$WAN_DIR are already
# fully populated, and a real dependency on network availability for zero
# benefit. Same skip-if-exists idiom as VANS_DIR/TWIFF_DIR above. Set
# FORCE_REDOWNLOAD_MODELS=1 to force a re-check/re-download anyway.
FORCE_REDOWNLOAD_MODELS="${FORCE_REDOWNLOAD_MODELS:-0}"
if [[ -d "$QWEN_DIR" ]] && [[ -n "$(ls -A "$QWEN_DIR" 2>/dev/null)" ]] && [[ "$FORCE_REDOWNLOAD_MODELS" != "1" ]]; then
    echo "QWEN_DIR=$QWEN_DIR already exists and is non-empty -- skipping Qwen2.5-VL-3B-Instruct download"
    echo "(set FORCE_REDOWNLOAD_MODELS=1 to force)."
else
    echo "Downloading Qwen/Qwen2.5-VL-3B-Instruct..."
    hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$QWEN_DIR"
fi
# SKIP_WAN_DOWNLOAD (added 2026-09-16, at explicit user request): Stage 0
# (train_stage0_latent_grounding.py, evaluate_stage0_latent_grounding.py)
# does NOT touch Wan/diffusers at all -- see that script's own module
# docstring. Wan only matters once Stage 1/2 exist, which they do not yet.
# The plain skip-if-exists check below is NOT enough on its own to keep
# WAN_DIR deleted: once you delete it to free space, the check below no
# longer has anything to skip and silently re-downloads it on the very next
# data_staging.sh run (e.g. a routine FORCE_RESTAGE_TRAIN=1 re-stage) --
# defeating the point of deleting it. This is an explicit, separate gate so
# deleting WAN_DIR actually stays deleted until you choose to bring it back.
# Default is 0 (unchanged behavior -- Wan downloads/skip-checks as before)
# so nothing changes unless you opt in.
SKIP_WAN_DOWNLOAD="${SKIP_WAN_DOWNLOAD:-0}"
if [[ "$SKIP_WAN_DOWNLOAD" == "1" ]]; then
    echo "SKIP_WAN_DOWNLOAD=1 -- not downloading/checking Wan2.1-T2V-1.3B this run, regardless of"
    echo "whether WAN_DIR currently exists. Unset this (or pass SKIP_WAN_DOWNLOAD=0) once Stage 1/2"
    echo "actually need it."
elif [[ -d "$WAN_DIR" ]] && [[ -n "$(ls -A "$WAN_DIR" 2>/dev/null)" ]] && [[ "$FORCE_REDOWNLOAD_MODELS" != "1" ]]; then
    echo "WAN_DIR=$WAN_DIR already exists and is non-empty -- skipping Wan2.1-T2V-1.3B download"
    echo "(set FORCE_REDOWNLOAD_MODELS=1 to force)."
else
    echo "Downloading Wan-AI/Wan2.1-T2V-1.3B..."
    hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$WAN_DIR"
fi

export METADATA_DIR
export TRAIN_SAMPLE_SIZE
export TRAIN_DATA_SOURCE

# SKIP-IF-EXISTS (added 2026-09-15, at explicit user request -- "i dont need
# to train on twiff again. I can use this completely"): staging is not free
# (loads/converts the full source dataset every time it runs) and re-running
# data_staging.sh for unrelated reasons -- e.g. just to pick up a val-split
# fix, or to re-chain download jobs -- shouldn't silently redo (and, for
# future_l1_50k, re-sort/re-truncate) training metadata you already have.
# Same idea as the VANS_DIR/TWIFF_DIR git-clone skip-if-exists checks above.
# Set FORCE_RESTAGE_TRAIN=1 to force a re-stage anyway (e.g. after changing
# TRAIN_SAMPLE_SIZE for the SAME METADATA_DIR path, which this check alone
# can't detect).
FORCE_RESTAGE_TRAIN="${FORCE_RESTAGE_TRAIN:-0}"
if [[ -d "$METADATA_DIR" ]] && [[ -n "$(ls -A "$METADATA_DIR" 2>/dev/null)" ]] && [[ "$FORCE_RESTAGE_TRAIN" != "1" ]]; then
    echo "METADATA_DIR=$METADATA_DIR already exists and is non-empty -- skipping training metadata"
    echo "staging (set FORCE_RESTAGE_TRAIN=1 to force a re-stage)."
elif [[ "$TRAIN_DATA_SOURCE" == "future_l1_50k" ]]; then
    echo "Staging Future-L1-50K (Eurayka/Future-L1-50K) -- Future-L1's own"
    echo "visual-gain-filtered subset of TwiFF, converted to TwiFF's column schema..."
    python -c '
import os
import re
from datasets import Dataset, load_dataset

output_dir = os.environ["METADATA_DIR"]
sample_size = os.environ.get("TRAIN_SAMPLE_SIZE") or None

ds = load_dataset("Eurayka/Future-L1-50K", split="train")
print(f"loaded Eurayka/Future-L1-50K: {len(ds)} rows, columns={ds.column_names}")

FRAME_MARKUP_RE = re.compile(r"(?m)^(frame_\d+|<image>)\s*\n?")

def extract_question_answer(conversations):
    human = next((t.get("value") for t in conversations if t.get("from") == "human"), None)
    gpt = next((t.get("value") for t in conversations if t.get("from") == "gpt"), None)
    if human is None or gpt is None:
        return None, None
    question = FRAME_MARKUP_RE.sub("", human).strip()
    return question, gpt

rows = []
n_bad_conv = 0
n_bad_meta = 0
for ex in ds:
    question, answer = extract_question_answer(ex["conversations"])
    if question is None or answer is None or not question or not answer:
        n_bad_conv += 1
        continue
    # CONFIRMED BUG, 2026-09-16 (real run, job 59104/59105 -- staged 50,000 rows,
    # chained download reported requested=0): this used to REPLACE ex["meta_data"]
    # entirely with a new dict of just source/class/gain/correct_in_8_base/
    # correct_in_8_hint, discarding the original meta_data wholesale. The original
    # TwiFF meta_data (and this datasets own, confirmed via the HF dataset viewer,
    # 2026-09-16 -- it carries the original TwiFF meta_data through unchanged) also
    # has start/end timestamp fields -- the ONLY thing download_panda70m_clips.py
    # (load_twiff_targets) reads to know what window to cut with ffmpeg. Discarding
    # them meant every one of 50,000 rows failed that check (if not start or not
    # end: skip), so the chained download job found zero usable targets and did
    # nothing -- not a network/bot-detection/cluster problem, a bug in this
    # conversion. FIX: preserve the original meta_data dict (Process, Summary,
    # classification, start, end) and ADD the extra fields to it instead of
    # replacing it wholesale.
    meta = dict(ex.get("meta_data") or {})
    if not meta.get("start") or not meta.get("end"):
        n_bad_meta += 1
        continue
    meta["source"] = "future_l1_50k"
    meta["gain"] = ex.get("gain")
    meta["correct_in_8_base"] = ex.get("correct_in_8_base")
    meta["correct_in_8_hint"] = ex.get("correct_in_8_hint")
    rows.append({
        "video": ex["video"],
        "question": question,
        "answer": answer,
        "question_images_index": ex["image"],
        "reasoning_images_index": ex["reasoning_image"],
        "meta_data": meta,
    })
if n_bad_conv:
    print(f"WARNING: {n_bad_conv} row(s) missing a usable human/gpt turn in conversations -- skipped")
if n_bad_meta:
    print(f"WARNING: {n_bad_meta} row(s) missing meta_data.start/end -- skipped (these rows would "
          f"make download_panda70m_clips.py fail to find a cut window; excluded here instead so "
          f"the count is visible now rather than silently producing requested=0 later)")
if not rows:
    raise RuntimeError(
        f"0 usable rows survived conversion out of {len(ds)} -- every row failed either the "
        f"conversations check ({n_bad_conv}) or the meta_data.start/end check ({n_bad_meta}). "
        f"This is the exact failure mode that produced a zero-target download job on 2026-09-16 -- "
        f"do not let this proceed silently."
    )

if sample_size is not None:
    n = int(sample_size)
    if n < len(rows):
        rows.sort(key=lambda r: (r["meta_data"]["gain"] if r["meta_data"]["gain"] is not None else -1), reverse=True)
        rows = rows[:n]
        print(f"TRAIN_SAMPLE_SIZE={n} < {len(ds)} -- kept the top {n} row(s) by gain (descending), "
              f"preserving the original visual-gain ranking instead of truncating arbitrarily")
    elif n > len(rows):
        print(f"WARNING: TRAIN_SAMPLE_SIZE={n} exceeds the {len(rows)} usable row(s) available -- "
              f"keeping all {len(rows)}")

print("Sample converted question (row 0):", repr(rows[0]["question"])[:200])
print("Sample converted answer   (row 0):", repr(rows[0]["answer"])[:200])

Dataset.from_list(rows).save_to_disk(output_dir)
print(f"Saved {len(rows)} row(s) (source=future_l1_50k) to {output_dir}")
'
else
    echo "Streaming the first $TRAIN_SAMPLE_SIZE TwiFF rows (train split)..."
    python -c '
import os
from itertools import islice
from datasets import Dataset, load_dataset
output_dir = os.environ["METADATA_DIR"]
n = int(os.environ["TRAIN_SAMPLE_SIZE"])
stream = load_dataset("Liu-Junhua/TwiFF-2.7M", split="train", streaming=True)
rows = list(islice(stream, n))
if len(rows) != n:
    raise RuntimeError(f"Expected {n} rows, received {len(rows)}")
Dataset.from_list(rows).save_to_disk(output_dir)
print(f"Saved {len(rows)} rows to {output_dir}")
'
fi

# VALIDATION SPLIT (added 2026-09-12, at explicit user request to stop
# evaluating Stage 0 on the same ~2000 rows it trains on): TwiFF-2.7M ships
# an official `validation` split on its own HF page (confirmed via the
# dataset card, not guessed) -- 1,451 rows, i.e. small enough to just save
# ALL of it rather than slicing, unlike the 2.71M-row `train` split above.
# This is a *different* split, not a subset of `train` -- so the clips it
# references still need their own download_panda70m_clips.sh /
# screen_panda70m_clip_quality.sh pass before train_stage0_latent_grounding.py
# can actually evaluate against it (see that script's own --val_metadata_dir
# flag and its module docstring, "Held-out validation"). UNCHANGED regardless
# of TRAIN_DATA_SOURCE -- see the TRAIN_DATA_SOURCE comment block above for why.
#
# CONFIRMED BUG, 2026-09-12 (job 58430 -- real staging run against this
# block, plain `load_dataset("Liu-Junhua/TwiFF-2.7M", split="validation")`):
# CastError, "column names don't match". The `train` split above loads fine
# because its parquet files' actual columns happen to match the dataset
# repo's DECLARED feature schema (the one huggingface `datasets` casts every
# row to). The `validation` split's parquet files do NOT match that declared
# schema -- they carry two extra columns (`question_images`, `reasoning_images`,
# the actual embedded image bytes for each row) that the declared schema
# doesn't list at all, so `datasets` refuses the cast outright rather than
# silently dropping columns. This is a real mismatch in how the dataset
# repo's `validation` parquet shards were produced vs. how its feature
# schema was declared -- not something wrong in this script's original
# `load_dataset(..., split="validation")` call, which is the normal/correct
# way to load a named split.
#
# FIX: bypass `datasets`' own schema-casting builder entirely by loading the
# validation split's raw parquet files directly through the generic
# "parquet" builder (which infers its schema from the files themselves,
# not from the repo's declared features), then drop the two extra
# image-bytes columns we don't need -- video frames are read later from the
# downloaded/cut mp4 clips (download_panda70m_clips.py), not from these
# embedded bytes. `huggingface_hub.HfApi.list_repo_files()` finds the exact
# val-*.parquet filenames rather than guessing a naming pattern.
FORCE_RESTAGE_VAL="${FORCE_RESTAGE_VAL:-0}"
if [[ -d "$VAL_METADATA_DIR" ]] && [[ -n "$(ls -A "$VAL_METADATA_DIR" 2>/dev/null)" ]] && [[ "$FORCE_RESTAGE_VAL" != "1" ]]; then
    echo "VAL_METADATA_DIR=$VAL_METADATA_DIR already exists and is non-empty -- skipping validation"
    echo "re-staging (set FORCE_RESTAGE_VAL=1 to force)."
else
echo "Streaming the full official TwiFF-2.7M validation split..."
export VAL_METADATA_DIR
python -c '
import os
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

output_dir = os.environ["VAL_METADATA_DIR"]
repo_id = "Liu-Junhua/TwiFF-2.7M"
KEEP_COLUMNS = ["video", "question", "answer", "question_images_index", "reasoning_images_index", "meta_data"]

api = HfApi()
files = api.list_repo_files(repo_id, repo_type="dataset")
val_files = sorted(f for f in files if os.path.basename(f).startswith("val-") and f.endswith(".parquet"))
if not val_files:
    raise RuntimeError(
        f"no val-*.parquet files found in {repo_id}s repo file listing "
        f"(first 20 files: {files[:20]}) -- the validation split may have been renamed/moved; "
        f"inspect the dataset repo on huggingface.co and update this filter."
    )
print(f"found {len(val_files)} validation parquet file(s): {val_files}")

local_paths = [hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f) for f in val_files]

# Loaded via the generic "parquet" builder, NOT
# load_dataset(repo_id, split="validation") -- the latter casts every row to
# the repo dataset cards DECLARED feature schema, which does not match this
# splits actual parquet columns and raises a hard CastError (confirmed
# 2026-09-12, job 58430s data_staging.sh run). Loading the raw parquet files
# directly infers the schema from the files themselves instead.
ds = load_dataset("parquet", data_files={"validation": local_paths}, split="validation")

extra_cols = [c for c in ds.column_names if c not in KEEP_COLUMNS]
if extra_cols:
    print(f"dropping {extra_cols} (not needed -- video frames come from the downloaded mp4 clips, "
          f"not these embedded image bytes)")
    ds = ds.remove_columns(extra_cols)
missing_cols = [c for c in KEEP_COLUMNS if c not in ds.column_names]
if missing_cols:
    raise RuntimeError(
        f"validation split is missing expected column(s) {missing_cols} after loading "
        f"(has {ds.column_names}) -- schema may differ from what this script assumes."
    )

ds.save_to_disk(output_dir)
print(f"Saved {len(ds)} validation rows to {output_dir}")
'
fi
echo "===== Data staging completed successfully ====="
echo "NOTE: train/validation source-video overlap has NOT been checked by this script (removed"
echo "2026-09-15 at explicit request -- it was gating AUTO_CHAIN_DOWNLOADS, which the user wants"
echo "to run before checking, not after). Run check_train_val_overlap.sh yourself once METADATA_DIR"
echo "and VAL_METADATA_DIR are both staged -- it's metadata-only (no clips needed), so it can run"
echo "any time, before or after the downloads finish."
echo "METADATA_DIR=$METADATA_DIR -- pass this SAME path to download_panda70m_clips.sh /"
echo "screen_panda70m_clip_quality.sh / train_stage0_latent_grounding.sh's own METADATA_DIR overrides."

# AUTO-CHAIN DOWNLOAD JOBS (added 2026-09-14, at explicit user request "add the
# validation clips download too in the same file"): metadata staging above is a
# short, lightweight, no-GPU job (--partition=medium, --time=04:00:00) --
# download_panda70m_clips.sh is a very different kind of job (yt-dlp/ffmpeg,
# --partition=long, --time=47:00:00, can run for many hours). Rather than
# literally pasting the downloader's logic into THIS script (which would force
# this job's time budget/partition to match the downloader's and turn a timeout
# mid-download into "resubmit the whole staging+download blob" instead of just
# the download step), this submits the two download jobs as SEPARATE SLURM
# jobs, chained via --dependency=afterok so they don't start until this
# staging job finishes successfully. Each keeps its own time budget and its
# own skip-if-exists resume behavior, unaffected by anything happening here.
#
# Set AUTO_CHAIN_DOWNLOADS=0 to disable this and go back to submitting
# download_panda70m_clips.sh yourself afterward (e.g. if you only want to
# restage metadata without kicking off multi-hour downloads every time).
AUTO_CHAIN_DOWNLOADS="${AUTO_CHAIN_DOWNLOADS:-1}"
if [[ "$AUTO_CHAIN_DOWNLOADS" == "1" ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "AUTO_CHAIN_DOWNLOADS=1 but this isn't running under SLURM (no SLURM_JOB_ID) --" >&2
        echo "skipping auto-chain (sbatch --dependency needs a real job ID). Submit" >&2
        echo "download_panda70m_clips.sh yourself for train/val." >&2
    else
        DOWNLOAD_SCRIPT="$SCRATCH/scripts/download_panda70m_clips.sh"
        if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
            echo "AUTO_CHAIN_DOWNLOADS=1 but $DOWNLOAD_SCRIPT not found -- skipping auto-chain." >&2
        else
            echo "Chaining train clip download (METADATA_DIR=$METADATA_DIR, OUTPUT_DIR=$DATA_DIR/panda70m_clips)..."
            METADATA_DIR="$METADATA_DIR" OUTPUT_DIR="$DATA_DIR/panda70m_clips" \
                sbatch --dependency=afterok:"$SLURM_JOB_ID" "$DOWNLOAD_SCRIPT"

            echo "Chaining validation clip download (METADATA_DIR=$VAL_METADATA_DIR, OUTPUT_DIR=$DATA_DIR/panda70m_clips_val)..."
            METADATA_DIR="$VAL_METADATA_DIR" OUTPUT_DIR="$DATA_DIR/panda70m_clips_val" \
                sbatch --dependency=afterok:"$SLURM_JOB_ID" "$DOWNLOAD_SCRIPT"

            echo "Both download jobs submitted with --dependency=afterok:$SLURM_JOB_ID -- they will"
            echo "start automatically once this staging job exits successfully. Check 'squeue -u \$USER'"
            echo "to see them queued (PD, dependency) right now. If THIS job fails, neither download"
            echo "job will ever start (afterok only fires on success) -- resubmit data_staging.sh."
        fi
    fi
else
    echo "AUTO_CHAIN_DOWNLOADS=0 -- not auto-submitting download jobs. Run download_panda70m_clips.sh"
    echo "yourself for train (METADATA_DIR=$METADATA_DIR) and validation (METADATA_DIR=$VAL_METADATA_DIR)."
fi

echo "End: $(date)"
