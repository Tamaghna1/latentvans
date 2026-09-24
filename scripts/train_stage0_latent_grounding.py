#!/usr/bin/env python
"""
Stage 0 -- Latent Grounding SFT for LatentVANS.

Trains Qwen2.5-VL-3B-Instruct to (a) answer questions about a video as
normal, and (b) along the way, emit an interleaved latent-reasoning span
(<|latent_start|> ... <|latent_end|>) whose hidden states are pulled toward
a frozen embedding of the FUTURE frames relevant to the answer -- without
ever showing the model those future frames as input. This is the actual
Stage 0 from claude/latentvans-research-plan.md, not the VLM->VDM bridge
smoke test (vlm_vdm_bridge_smoke_test.py) (which only checks that the
VLM->Proj->Wan plumbing doesn't crash on synthetic tensors). Stage 0 does
not touch Wan/diffusers at all -- it is VLM-only, per the plan.

Loss (matches the real Future-L1 paper, arXiv:2606.05769, which TwiFF-2.7M
is that paper's own training set for):
    L_latent = (1/|S|) sum_{t in S} || h_t - e*_t ||^2_2
    L_SFT    = L_CE + lambda * L_latent      (lambda = 0.1, Future-L1's value)
where h_t is the VLM's own final-layer hidden state at a <|latent_pad|>
position, and e*_t is a target embedding produced by Qwen2.5-VL-3B's OWN
vision tower (`vlm.visual`) run on the future/reasoning frames -- NOT an
external CLIP/DINO encoder. Future-L1 explicitly does this (Sec 3.2) so
that h_t and e*_t already live in the same 2048-dim space (Qwen2.5-VL-3B's
vision tower is built to project into the language model's hidden_size,
which is exactly why it can splice vision tokens into the token stream in
the first place) -- no NEW/learned projection layer added by this script,
unlike the Proj module in the VLM->VDM bridge smoke test
(vlm_vdm_bridge_smoke_test.py) (that one bridges into Wan's 4096-dim space
for Stage 1/2, and has no role here). CORRECTION (2026-09-09, confirmed by
a real training crash -- see compute_future_embedding()'s own comment):
"no extra projection" is true architecturally (the merger that does this
projection is already part of the pretrained vision tower, not something
this script adds), but calling `vision_tower(pixel_values, grid_thw=...)`
directly does NOT apply that merger in this transformers version -- it
returns the raw, PRE-merger 1280-dim ViT patch representation (1280 is
the ViT width shared across all Qwen2.5-VL sizes: 2B/3B/7B/72B; each
checkpoint's own `merger` submodule is what actually projects up to that
specific model's hidden_size). compute_future_embedding() now applies
`vision_tower.merger` explicitly when the raw output doesn't already
match QWEN_HIDDEN_SIZE, and a startup preflight check (verify_vision_
target_dim(), called from main() right after the model loads) fails loud
with a clear diagnostic if that still doesn't produce the expected width,
rather than crashing deep inside F.mse_loss after however many minutes
were already spent loading the model and starting the run.

DESIGN DECISION worth knowing about (not dictated by either paper, since
TwiFF's annotation granularity doesn't map 1:1 onto Future-L1's original
per-position single-frame target): Future-L1 aligns each of its L_max
latent positions to its OWN distinct future frame. Our research plan's
extension pools a short K-frame window per position instead of one static
frame, to carry motion. TwiFF-2.7M gives one flat list of "reasoning"
frame indices per example (1-7 frames), not a pre-split-by-position list,
so this script pools ALL of an example's reasoning frames into ONE target
embedding and points every latent position in that example's span at the
same pooled target, rather than trying to invent a per-position split from
data that doesn't naturally carry one. This is a reasonable, clearly-flagged
simplification, not the only valid reading of the plan -- revisit if you
have a reason to want distinct per-position targets once you've looked at
real batches of TwiFF data.

LoRA fine-tuning (added 2026-09-07, now the default -- see --use_lora)
------------------------------------------------------------------------
The Panda-70M clip pipeline (see claude/latentvans-handoff.md) currently
produces on the order of ~1800-1900 usable clips -- still small relative
to the research plan's VANS-Data-100K/TwiFF-2.7M-scale assumptions, so
LoRA stays the default rather than full-parameter fine-tuning of all 3B
weights (the default --max_steps 2000 * effective batch 8 = 16,000 example
draws, which is several passes over ~1800 clips -- much healthier than the
original ~150-clip pilot's 50-100+ passes, but still small enough that
LoRA's reduced capacity/overfitting risk is the safer default). LoRA cuts
the trainable parameter count by orders of magnitude, which is both the
standard fix for small-data fine-tuning and meaningfully cheaper on GPU
memory. What's actually trainable now:
  - Low-rank adapters on the LANGUAGE MODEL backbone's attention projections
    (q_proj/k_proj/v_proj/o_proj by default -- see --lora_target_modules;
    kept to attention-only, not also the MLP gate/up/down projections, as a
    conservative default given how little data there currently is; revisit
    once a real-scale Panda-70M download exists).
  - `embed_tokens` and `lm_head`, kept FULLY trainable (not LoRA-adapted) via
    peft's `modules_to_save` -- the three <|latent_start|>/<|latent_end|>/
    <|latent_pad|> tokens are brand new vocab rows with no pretrained weight
    for a low-rank delta to adapt; they need real gradient updates from
    scratch, which only a fully-trainable copy of these two layers provides.
    NOTE (2026-09-12): this is also precisely why train loss reaching near-
    zero early in a run is not, by itself, evidence the latent-grounding
    objective generalizes -- these two fully-trainable layers have more
    than enough free capacity to key off which specific (repeatedly-shown)
    clip is currently being conditioned on. See "Held-out validation" below.
What's still frozen either way: the vision tower (`vlm.visual`) -- it's only
used to compute the frozen target embedding this script's loss depends on,
never trained, in both the LoRA and full-FT paths. A defensive check
explicitly re-freezes anything under the vision tower's parameter names in
case `target_modules`' name-based matching (e.g. "q_proj") happens to also
match a same-named module inside the ViT itself -- not confirmed either way
for Qwen2.5-VL's exact vision-tower module names, so this is cheap insurance
rather than a fix for a confirmed problem.

Practical differences this causes: `--lr` now defaults to 2e-4 (a typical
LoRA learning rate for freshly-initialized low-rank adapters) rather than
the 1e-5 used for full fine-tuning -- if you pass --no_lora, lower --lr back
down, 2e-4 would likely be too aggressive for updating all 3B weights
directly. Checkpoints are now much smaller: `vlm.save_pretrained()` on a
PeftModel writes only the adapter weights plus the two `modules_to_save`
layers, not a full copy of the 3B-parameter base model. This also means a
saved Stage 0 checkpoint is no longer a standalone model -- consuming it
later (e.g. from a Stage 1 script, not yet written) means loading the base
Qwen2.5-VL-3B weights fresh and then attaching the adapter via
`peft.PeftModel.from_pretrained(base_model, checkpoint_dir)`, not a plain
`from_pretrained(checkpoint_dir)` on the checkpoint alone. Flagging this now
since it will matter the moment Stage 1 exists, not implementing a
merge-and-save convenience yet since nothing downstream needs it today.

Full fine-tuning is still available via --no_lora for comparison or once
the dataset is large enough that LoRA's capacity ceiling becomes the
limiting factor rather than data volume -- that code path is unchanged from
before this rewrite.

New-token embedding initialization (added 2026-09-24, --mean_resizing)
------------------------------------------------------------------------
The three latent-reasoning tokens (<|latent_start|>/<|latent_end|>/
<|latent_pad|>) are brand new vocab rows added via
tokenizer.add_special_tokens() + vlm.resize_token_embeddings(). How those
new rows get their INITIAL values matters: transformers' default behavior
for resize_token_embeddings (mean_resizing=True, the actual default since
~4.46) draws each new row from a multivariate normal fit to the mean/
covariance of the EXISTING embedding matrix, rather than small independent
random noise -- documented by HF to reduce next-token-distribution KL-
divergence for causal LMs versus naive random init (see that method's own
docstring, citing https://nlp.stanford.edu/~johnhew/vocab-expansion.html).
This script now passes mean_resizing=True EXPLICITLY (redundant with the
current default, but self-documenting and safe against
huggingface/transformers#35357, which proposes flipping that default back
to False for large-embedding-dim models like Qwen2 on performance
grounds -- if that lands in a future transformers release, this script's
behavior won't silently change underneath it).

Separately, there is a REAL, CONFIRMED transformers bug worth knowing about
(huggingface/transformers#35141, confirmed against transformers 4.46.3,
supposedly fixed by transformers PRs #45079/#36221): when a model's
tie_word_embeddings is False (input embed_tokens and output lm_head are
separate weight matrices -- not confirmed either way for Qwen2.5-VL-3B from
this session, no cluster access to check `vlm.config.tie_word_embeddings`
directly), resize_token_embeddings() can create a new output-embedding
Linear module that a LATER post_init() call silently RE-initializes back to
small random noise -- undoing mean_resizing's careful init on the output
side specifically, with no error or warning. Whether the transformers
version actually installed on this cluster (required: >=4.49.0, per the
ImportError message below) still has this bug is NOT independently
confirmed here. Rather than guess, log_new_token_embedding_norms() (called
right after resize_token_embeddings() in load_model_and_tokenizer()) prints
the actual L2 norm of each new token's row in both the input and (if
untied) output embeddings, next to a sample of existing rows' mean norm --
a new-token row near-zero relative to that sample is the signature of this
bug (or of some other silent re-init); check the job log for that WARNING
line before trusting a run's early training behavior, and if you see it,
check `pip show transformers` against the two PR numbers above.

Held-out validation (added 2026-09-12, --val_metadata_dir)
------------------------------------------------------------------------
Every run of this script before today trained AND was only ever read via
its own training loss -- there was no held-out data anywhere in this
project's pipeline. That is a real gap, not a minor one: job 57910 (the
first Stage 0 run to complete 2000 steps) showed train loss collapsing to
near-zero by roughly step 150-200 and staying flat for the rest of the
run, which is exactly the signature you'd expect from a LoRA adapter +
fully-trainable embed_tokens/lm_head memorizing a small, heavily-repeated
clip pool (~1850 quality-screened clips, drawn with replacement roughly
15-20x each over a 2000-step/effective-batch-8 run) -- and with no held-out
signal, there was and is no way to tell that apart from the model actually
learning a generalizable latent-grounding mapping.

TwiFF-2.7M ships an official `validation` split on its own HF dataset page
(1,451 rows, confirmed via the dataset card) -- a genuinely different set
of rows from the 2,000-row `train` slice this project stages, not a
resplit of it. data_staging.sh now also streams and saves that full
validation split (see its own comment block). Because it references a
different set of source videos, its clips need their own
download_panda70m_clips.sh / screen_panda70m_clip_quality.sh pass before
they can be used here -- see both scripts' own comment blocks for the
env-var overrides (METADATA_DIR/OUTPUT_DIR, METADATA_DIR/CLIPS_DIR) that
point them at the validation slice instead of train's.

Once validation clips exist, pass --val_metadata_dir (and, if the clips
live somewhere other than --video_root, --val_video_root) to turn on
periodic held-out evaluation: every --eval_every steps (defaults to
--save_every if unset), the training loop switches the model to eval()
mode, runs up to --eval_max_examples (default 200) examples from the
validation stream under torch.no_grad(), computes the same loss_ce/
loss_latent this script already tracks for training, logs them to
metrics.jsonl as val_loss_ce/val_loss_latent/val_loss_total (and to W&B if
--wandb_project is set), then switches back to train() mode. This never
backpropagates through validation data and never influences the optimizer
or LR schedule -- it is read-only monitoring. If val loss tracks train loss
down, that is real evidence of generalization; if val loss stays flat or
rises while train loss keeps falling, that is direct evidence of
memorization -- something train loss alone cannot distinguish. If
--val_metadata_dir is not passed, this script behaves exactly as before
(no eval, a warning logged once at startup) -- this is opt-in, not a
required flag, since a given run may not yet have validation clips staged.

Pre-extracted frame cache (added 2026-09-24, --frames_root/--val_frames_root)
------------------------------------------------------------------------
TwiFFStage0Stream.next_example() previously always live-decoded the FULL
raw clip via decord on every single draw, twice (question_images_index +
reasoning_images_index), even though each clip only ever contributes a
handful of frames to any example. claude/latentvans-handoff.md item 12
documents a sustained ~30-41% decord decode-failure rate across every real
run so far -- a clip that fails to decode fails EVERY time it's drawn, so
this cost was being re-paid on every step of a multi-hour run instead of
being discovered once, and full clips were sitting on disk-quota-
constrained scratch when only a few frames per clip are ever read (this
project's cluster account has hit OSError: Disk quota exceeded twice,
once fatally mid-training-run, as of 2026-09-24).

extract_stage0_frames.py (new, 2026-09-24) pre-extracts exactly the frames
this script needs (the union of question_images_index and
reasoning_images_index per clip) into
<frames_root>/<clip_stem>/<frame_idx>.jpg, once, ahead of training. Pass
--frames_root (for the training stream) and/or --val_frames_root (for the
validation stream, independently) to have TwiFFStage0Stream read from that
cache instead of live-decoding raw video for that stream. This is
OPT-IN and fully backward compatible: leaving both unset reproduces the
exact prior behavior (live decord decode from --video_root/
--val_video_root) with no other change. --video_root is still required
either way -- it remains the canonical reference for which clip a
metadata row names, and the fallback path when --frames_root is unset.

IMPORTANT -- this does NOT fix the underlying decord decode-failure rate.
A clip extract_stage0_frames.py couldn't decode still isn't available here
either (it will log as a skip the same way a missing/corrupt raw clip
always has) -- see that script's own module docstring, "WHAT THIS SCRIPT
DOES NOT DO", for the fix that actually reduces the failure rate
(download_panda70m_clips.py's decord_can_read()-validated cut_clip(), plus
--recut_from_list to apply it only to currently-broken clips -- see that
script's own module docstring, "TARGETED RECUT"). Also note: switching a
stream to --frames_root only covers clips extract_stage0_frames.py has
actually processed for the --metadata_dir currently in use -- if you
restage a larger/different metadata slice later, the newly-added rows'
frames won't exist in the cache until extract_stage0_frames.py is re-run
against them.

Weights & Biases logging (added 2026-09-09, --wandb_project, OPT-IN)
------------------------------------------------------------------------
This project's own handoff doc (claude/latentvans-handoff.md) has said "No
Weights & Biases" since early on -- that was about not silently depending
on an external service nothing else in this project used. The user now has
a W&B account and explicitly asked to see the loss curves live, so this is
a deliberate, explicit reversal for this one script, not a project-wide
default: W&B logging is entirely OFF unless --wandb_project is passed.
When it's off, nothing changes -- no new import is even attempted.
When it's on: every logged step (the same cadence as --log_every, which
already writes metrics.jsonl) also calls wandb.log() with loss_ce,
loss_latent, the combined weighted loss, grad_norm, lr, and skipped_total.
metrics.jsonl keeps being written exactly as before either way -- W&B is
an additional, disposable view, not the source of truth for these numbers
(if the compute node's network hiccups mid-run, training does not stop:
see the try/except below). Requires being logged in already (`wandb login`
once on the login node, which persists to ~/.netrc and is picked up from
compute nodes too since they share $HOME) or a WANDB_API_KEY in the job's
environment -- this script does not accept an API key as a CLI argument on
purpose (CLI args land in job logs and `ps` output; a wandb credential
should not).

Quality-screen filtering (added 2026-09-09, --exclude_flagged_json)
------------------------------------------------------------------------
download_panda70m_clips.py's clips have since been quality-screened by
screen_panda70m_clip_quality.py (a Qwen2.5-VL-3B-Instruct zero-shot judge --
see that script's module docstring and claude/latentvans-handoff.md item 5
for the full story, including a confirmed stale-file bug in the downloader
that has since been fixed and re-verified). That screen produces
`quality_screen_flagged.json` under the clips directory: every clip the
judge called MISMATCH, PARSE_ERROR, or NO_METADATA. Pass that file via
--exclude_flagged_json and TwiFFStage0Stream will drop every TwiFF row
whose `video` filename appears in it, so this run never draws on a clip
the screen flagged as suspect. This is an exclusion list, not a guarantee
of correctness for what's left -- the screen is a heuristic (documented to
have real false negatives; see the handoff doc) and manually eyeballing a
sample of the surviving MATCH clips is still worth doing, just not a
blocker for starting this run per explicit instruction to exclude the
flagged samples and proceed. --val_exclude_flagged_json is the same idea,
applied to the validation stream, from validation's own quality screen.
NOTE (2026-09-24): this has NOT been run against Future-L1-50K -- see
claude/latentvans-training-decisions-log.md's 2026-09-23 entry, "quality
screen explicitly skipped", a real flagged risk carried into any run
against that metadata slice, not silently accepted.

PREREQUISITE STATUS (updated 2026-09-24): Future-L1-50K clips are
downloaded (partial coverage as of this update -- see
claude/latentvans-training-decisions-log.md for the current downloaded/
decode-clean/missing counts, which are actively changing as recut/download
jobs run) and NOT quality-screened (see note above). --video_root should
point at the populated clips directory. The NoUsableExamplesError below
exists for the case where it still doesn't (wrong path, clips not yet
staged on this node's scratch, etc.), not because clips are known to be
missing.

Data: expects the staged metadata slice (data_staging.sh) at
$DATA_DIR/<slice> (a datasets.Dataset saved via save_to_disk, columns:
video, question, answer, question_images_index, reasoning_images_index,
meta_data -- confirmed from the dataset's own card, not guessed). This
schema is shared by both the original TwiFF-2.7M `twiff_metadata_2000`
slice and the newer Future-L1-50K `future_l1_50k_metadata_full` slice --
--metadata_dir picks which one a given run actually trains on. Validation
uses the same schema, staged separately at $DATA_DIR/twiff_metadata_val
(see "Held-out validation" above) -- this stays TwiFF's official
validation split regardless of which slice --metadata_dir points at (see
check_train_val_overlap.sh and the 2026-09-23 decision log entry
confirming zero/negligible overlap against Future-L1-50K specifically).

Usage
-----
    python train_stage0_latent_grounding.py \\
        --qwen_dir /scratch/users/anirban/tamaghnam/latentvans/checkpoints/Qwen2.5-VL-3B-Instruct \\
        --metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/future_l1_50k_metadata_full \\
        --video_root /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips \\
        --output_dir /scratch/users/anirban/tamaghnam/latentvans/checkpoints/stage0_futurel150k_run1 \\
        --val_metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/twiff_metadata_val \\
        --val_video_root /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips_val \\
        --max_steps 6000 --batch_size 1 --grad_accum_steps 8 \\
        --wandb_project latentvans-stage0

    (LoRA is on by default. To override its hyperparameters:)
        --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \\
        --lora_target_modules q_proj,k_proj,v_proj,o_proj

    (Or to fully fine-tune all 3B parameters instead, as before this rewrite:)
        --no_lora --lr 1e-5

    (To read pre-extracted frames instead of live-decoding raw video, once
    extract_stage0_frames.py has been run -- see "Pre-extracted frame
    cache" above:)
        --frames_root /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_frames \\
        --val_frames_root /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_frames_val

Needs `decord` for frame-accurate video seeking and `peft` for LoRA (both
pip install decord peft) -- not yet in any install list; added to this
script's own dependency check. `decord` is only imported when at least one
active stream is still live-decoding raw video (i.e. --frames_root/
--val_frames_root isn't covering every stream in use) -- see main().
"""
import argparse
import json
import math
import os
import random
import sys
import time
import traceback

import torch
import torch.nn.functional as F


LATENT_START = "<|latent_start|>"
LATENT_END = "<|latent_end|>"
LATENT_PAD = "<|latent_pad|>"

QWEN_HIDDEN_SIZE = 2048  # Qwen2.5-VL-3B-Instruct text hidden_size (confirmed via config.json)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dtype_from_str(s):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def load_excluded_videos(exclude_flagged_json):
    """Reads a quality_screen_flagged.json (screen_panda70m_clip_quality.py's output --
    a JSON array of report entries, each with a "video" filename key) and returns the
    set of filenames to exclude from training. Returns an empty set if the arg is None."""
    if not exclude_flagged_json:
        return set()
    with open(exclude_flagged_json) as f:
        flagged = json.load(f)
    videos = {entry["video"] for entry in flagged if "video" in entry}
    log(f"loaded {len(videos)} flagged video filename(s) to exclude from {exclude_flagged_json}")
    return videos


class NoUsableExamplesError(RuntimeError):
    """Raised when far too many consecutive examples were skipped (missing
    video file / out-of-range frame index / corrupt clip) -- almost always
    means --video_root doesn't actually have the Panda-70M clips yet, and
    training should stop with a clear message instead of spinning forever."""


def load_video_frames(video_path, indices):
    """Extract specific 0-indexed frame numbers from an mp4 as PIL Images.
    Uses decord for direct frame-index seeking (no fps/pts bookkeeping).
    Only used for a stream when --frames_root/--val_frames_root is NOT
    set for it -- see load_extracted_frames() for the pre-extracted-cache
    path, and module docstring, "Pre-extracted frame cache"."""
    import decord
    from PIL import Image

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    n = len(vr)
    valid = [i for i in indices if 0 <= i < n]
    if not valid:
        raise ValueError(f"no valid frame indices in {video_path} (has {n} frames, wanted {indices})")
    batch = vr.get_batch(valid).asnumpy()  # (len(valid), H, W, 3) uint8
    return [Image.fromarray(f) for f in batch]


def load_extracted_frames(frames_root, video_filename, indices):
    """Loads pre-extracted frame JPEGs written by extract_stage0_frames.py
    (<frames_root>/<clip_stem>/<frame_idx>.jpg) instead of live-decoding
    the raw clip. Mirrors load_video_frames's signature/return type
    exactly (a list of PIL Images in the same order as `indices`), so
    every downstream caller (build_training_example, compute_future_
    embedding) needs zero changes -- they consume "context_frames"/
    "future_frames" without caring where the frames actually came from.
    See module docstring, "Pre-extracted frame cache". Raises (rather
    than silently skipping) on a missing/corrupt frame file -- the caller
    (TwiFFStage0Stream.next_example) already catches this the same way it
    already catches a load_video_frames failure."""
    from PIL import Image

    stem = video_filename[:-4] if video_filename.endswith(".mp4") else video_filename
    clip_dir = os.path.join(frames_root, stem)
    if not indices:
        raise ValueError(f"no frame indices requested for {video_filename} (indices={indices})")
    frames = []
    for i in indices:
        path = os.path.join(clip_dir, f"{i}.jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no extracted frame at {path} (wanted indices {indices} for {video_filename}) -- "
                f"has extract_stage0_frames.py been run against this clip yet?"
            )
        frames.append(Image.open(path).convert("RGB"))
    return frames


class TwiFFStage0Stream:
    """A plain shuffled-with-replacement stream over the staged TwiFF slice,
    not a torch.utils.data.Dataset -- deliberately, because a bad/missing
    Panda-70M clip must be SKIPPED, not crash a multi-hour training run, and
    the standard DataLoader/collate machinery doesn't make that easy when
    each example also needs its own variable-length video decode. Call
    next_example() in a loop; it returns None for a skipped row so callers
    can just try again. Used for BOTH the training stream (metadata_dir=the
    staged `train` slice) and, since 2026-09-12, the held-out validation
    stream (metadata_dir=the staged `validation` slice, a disjoint set of
    TwiFF rows/source videos) -- same class, different data, so validation
    reuses the exact same skip-handling/exclusion logic training already
    relies on rather than a parallel reimplementation.

    frames_root (added 2026-09-24, optional): when set, frames are read
    from extract_stage0_frames.py's pre-extracted JPEG cache instead of
    live-decoding the raw clip via decord -- see module docstring,
    "Pre-extracted frame cache". Defaults to None, which reproduces the
    exact prior (always-live-decode) behavior -- existing callers that
    don't pass this keyword (e.g. evaluate_stage0_latent_grounding.py, as
    of 2026-09-24) are unaffected."""

    def __init__(self, metadata_dir, video_root, seed=0, exclude_videos=None, frames_root=None):
        from datasets import load_from_disk

        self.rows = load_from_disk(metadata_dir)
        required_cols = {"video", "question", "answer", "question_images_index", "reasoning_images_index"}
        missing = required_cols - set(self.rows.column_names)
        if missing:
            raise ValueError(
                f"metadata_dir={metadata_dir} is missing expected TwiFF-2.7M columns {missing} "
                f"(has {self.rows.column_names}) -- schema may have changed, or this isn't the "
                f"TwiFF slice data_staging.sh produced."
            )
        self.video_root = video_root
        self.frames_root = frames_root  # set = read extract_stage0_frames.py's cache, skip decord entirely
        self.rng = random.Random(seed)

        exclude_videos = exclude_videos or set()
        all_indices = list(range(len(self.rows)))
        if exclude_videos:
            video_col = self.rows["video"]
            self.order = [i for i in all_indices if video_col[i] not in exclude_videos]
            n_excluded_rows = len(all_indices) - len(self.order)
            log(f"quality-screen exclusion: {n_excluded_rows} of {len(all_indices)} TwiFF row(s) "
                f"reference a flagged clip and will be skipped this run "
                f"({len(self.order)} row(s) remain usable)")
        else:
            self.order = all_indices
        self.rng.shuffle(self.order)
        self.pos = 0
        self.n_skipped_total = 0

    def __len__(self):
        return len(self.order)

    def next_example(self):
        if self.pos >= len(self.order):
            self.rng.shuffle(self.order)
            self.pos = 0
        row = self.rows[self.order[self.pos]]
        self.pos += 1
        video_filename = row["video"]

        if self.frames_root:
            # Pre-extracted-frame-cache path (2026-09-24) -- see module docstring, "Pre-extracted
            # frame cache". Cheap directory-existence pre-check mirrors the raw-video path's
            # os.path.exists check below: a clip extract_stage0_frames.py hasn't reached yet is a
            # silent, uncounted-as-noisy skip (this is the expected state for any clip not yet
            # processed by a partially-run extraction job), not a logged error.
            stem = video_filename[:-4] if video_filename.endswith(".mp4") else video_filename
            clip_dir = os.path.join(self.frames_root, stem)
            if not os.path.isdir(clip_dir):
                self.n_skipped_total += 1
                return None
            try:
                context_frames = load_extracted_frames(self.frames_root, video_filename, row["question_images_index"])
                future_frames = load_extracted_frames(self.frames_root, video_filename, row["reasoning_images_index"])
            except Exception as e:
                log(f"  skipping {video_filename}: {e}")
                self.n_skipped_total += 1
                return None
        else:
            # Original live-decode path, unchanged.
            video_path = os.path.join(self.video_root, video_filename)
            if not os.path.exists(video_path):
                self.n_skipped_total += 1
                return None
            try:
                context_frames = load_video_frames(video_path, row["question_images_index"])
                future_frames = load_video_frames(video_path, row["reasoning_images_index"])
            except Exception as e:
                log(f"  skipping {video_filename}: {e}")
                self.n_skipped_total += 1
                return None

        return {
            "video": video_filename,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "question": row["question"],
            "answer": row["answer"],
        }

    def next_usable_example(self, max_consecutive_skips=200):
        consecutive_skips = 0
        while True:
            ex = self.next_example()
            if ex is not None:
                return ex
            consecutive_skips += 1
            if consecutive_skips >= max_consecutive_skips:
                raise NoUsableExamplesError(
                    f"{consecutive_skips} consecutive examples were unusable (missing video file / "
                    f"bad frame index / corrupt clip). This almost certainly means --video_root="
                    f"{self.video_root!r} does not contain the Panda-70M clips TwiFF-2.7M's `video` "
                    f"column names -- {self.n_skipped_total} skipped out of {self.pos} examined so "
                    f"far this pass. Fix --video_root (or fetch the clips) before training."
                )


def get_vision_tower(vlm):
    for path in ("visual", "model.visual"):
        obj = vlm
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(
        "Could not find the vision tower at vlm.visual or vlm.model.visual -- transformers may have "
        "renamed it in this version. Check `[n for n, _ in vlm.named_children()]` and update "
        "get_vision_tower()."
    )


@torch.no_grad()
def compute_future_embedding(future_frames, processor, vision_tower, device, compute_dtype):
    """One 2048-dim target per example: run Qwen2.5-VL's OWN vision tower on
    each future/reasoning frame independently, mean-pool each frame's patch
    tokens to one vector, then mean over frames (the "pool a K-frame window"
    extension from the research plan -- see module docstring for how K is
    chosen here from TwiFF's own annotation).

    MERGER FIX (2026-09-09): calling vision_tower(pixel_values, grid_thw=...)
    alone returns the RAW pre-merger ViT patch representation (1280-dim, the
    width shared by all Qwen2.5-VL sizes), not the 2048-dim representation
    the 3B model's own merger produces to splice into its text stream --
    confirmed by a real crash (RuntimeError: 1280 vs 2048 at F.mse_loss on
    the very first training step). If the raw output doesn't already match
    QWEN_HIDDEN_SIZE, explicitly apply vision_tower.merger to it -- this is
    a no-op (never triggers) on any transformers version where forward()
    already includes the merger step internally.
    """
    per_frame_vecs = []
    for frame in future_frames:
        proc_out = processor(images=[frame], return_tensors="pt")
        pixel_values = proc_out["pixel_values"].to(device=device, dtype=compute_dtype)
        grid_thw = proc_out["image_grid_thw"].to(device)
        patch_embeds = vision_tower(pixel_values, grid_thw=grid_thw)
        if hasattr(patch_embeds, "last_hidden_state"):
            patch_embeds = patch_embeds.last_hidden_state
        if patch_embeds.shape[-1] != QWEN_HIDDEN_SIZE and hasattr(vision_tower, "merger"):
            patch_embeds = vision_tower.merger(patch_embeds)
        per_frame_vecs.append(patch_embeds.float().mean(dim=0))  # (2048,)
    target = torch.stack(per_frame_vecs, dim=0).mean(dim=0)  # (2048,)
    return target


def verify_vision_target_dim(processor, vision_tower, device, compute_dtype):
    """One dry run on a synthetic image, called once from main() right after
    the model loads, to confirm compute_future_embedding() actually produces
    a QWEN_HIDDEN_SIZE-dim target BEFORE spending any time on the real TwiFF
    data loading and training loop. Added 2026-09-09 after a real run got all
    the way to the first training step (model loaded, W&B run started) before
    crashing on this exact mismatch -- fail loud and immediately here instead,
    with a diagnostic pointing straight at the cause, if a future transformers
    or model version changes vision-tower behavior again."""
    from PIL import Image

    dummy = Image.new("RGB", (280, 280))  # multiple of patch_size(14)*merge_size(2); safe default grid
    target = compute_future_embedding([dummy], processor, vision_tower, device, compute_dtype)
    if target.shape[-1] != QWEN_HIDDEN_SIZE:
        log(f"FATAL: vision-tower target embedding is {target.shape[-1]}-dim, expected "
            f"{QWEN_HIDDEN_SIZE} (Qwen2.5-VL-3B's text hidden_size). compute_future_embedding() "
            f"already applies vision_tower.merger() as a fallback when the raw output doesn't "
            f"match -- if you're seeing this, that fallback didn't fix it either. Inspect "
            f"`[n for n, _ in vision_tower.named_children()]` and the actual object/shape returned "
            f"by vision_tower(...) directly to find the real merge/projection step for whatever "
            f"transformers version is installed, and update compute_future_embedding() accordingly.")
        sys.exit(1)
    log(f"vision-tower target embedding dim confirmed: {target.shape[-1]} (matches QWEN_HIDDEN_SIZE)")


def log_new_token_embedding_norms(vlm, tokenizer, new_token_strs):
    """Diagnostic, not a fix -- added 2026-09-24 (see module docstring, 'New-token embedding
    initialization'). Logs the L2 norm of each newly-added special token's row in both the input
    embeddings (embed_tokens) and, if untied from input embeddings, the output embeddings
    (lm_head), right after resize_token_embeddings(..., mean_resizing=True) -- and compares them
    against a sample of PRE-EXISTING rows' norms.

    This exists because of a real, confirmed transformers bug (huggingface/transformers#35141,
    confirmed against 4.46.3): when tie_word_embeddings=False, a LATER post_init() call can
    silently re-initialize the just-resized output embeddings back to small random noise, undoing
    mean_resizing entirely -- no error, no warning, just wrong numbers from then on. Whether this
    project's installed transformers version still has that bug is NOT independently confirmed
    here (no cluster access from this session) -- this prints the actual norms into the job log so
    it's visible directly instead of assumed either way.

    A new-token row with a norm wildly smaller than the existing-row sample mean is the signature
    to watch for. This is read-only logging -- it never modifies the embeddings itself.
    """
    input_emb = vlm.get_input_embeddings().weight
    output_emb_module = vlm.get_output_embeddings()
    tied = bool(getattr(getattr(vlm, "config", None), "tie_word_embeddings", False))

    n_new = len(new_token_strs)
    vocab_size = input_emb.shape[0]
    sample_start = max(0, vocab_size - n_new - 1000)
    sample_end = max(sample_start, vocab_size - n_new)
    existing_sample_ids = list(range(sample_start, sample_end))
    if not existing_sample_ids:
        log("  (skipping new-token embedding norm check -- vocab too small to sample existing rows)")
        return
    existing_mean_norm = input_emb[existing_sample_ids].float().norm(dim=-1).mean().item()

    log(f"  new-token embedding norms (existing-row sample mean norm={existing_mean_norm:.3f}, "
        f"tie_word_embeddings={tied}):")
    for tok in new_token_strs:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        in_norm = input_emb[tok_id].float().norm().item()
        line = f"    {tok!r} (id={tok_id}): input norm={in_norm:.3f}"
        suspect = in_norm < existing_mean_norm * 0.1
        if not tied and output_emb_module is not None and hasattr(output_emb_module, "weight"):
            out_norm = output_emb_module.weight[tok_id].float().norm().item()
            line += f", output(lm_head) norm={out_norm:.3f}"
            if out_norm < existing_mean_norm * 0.1:
                suspect = True
        if suspect:
            line += ("  <-- WARNING: looks near-zero vs existing rows, not what mean_resizing "
                     "should produce -- see this function's own docstring (transformers#35141)")
        log(line)


def build_training_example(example, processor, tokenizer, latent_span_len, max_answer_chars):
    answer = example["answer"]
    if len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars]

    latent_span = LATENT_PAD * latent_span_len
    messages_prompt = [
        {"role": "system", "content": "You are a helpful assistant that reasons about videos."},
        {"role": "user", "content": (
            [{"type": "image"} for _ in example["context_frames"]]
            + [{"type": "text", "text": example["question"]}]
        )},
    ]
    prompt_text = processor.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    full_messages = messages_prompt + [
        {"role": "assistant", "content": f"{LATENT_START}{latent_span}{LATENT_END} {answer}"},
    ]
    full_text = processor.apply_chat_template(full_messages, tokenize=False)

    # Tokenize the prompt alone (to get its length for label masking) and the
    # full sequence together, both through the *same* processor call style as
    # the VLM->VDM bridge smoke test (vlm_vdm_bridge_smoke_test.py), so
    # image-token expansion is handled identically.
    prompt_inputs = processor(text=[prompt_text], images=example["context_frames"], return_tensors="pt")
    full_inputs = processor(text=[full_text], images=example["context_frames"], return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100  # only supervise the assistant turn (latent span + answer)
    full_inputs["labels"] = labels

    latent_pad_id = tokenizer.convert_tokens_to_ids(LATENT_PAD)
    latent_mask = full_inputs["input_ids"][0] == latent_pad_id
    n_latent = int(latent_mask.sum().item())
    if n_latent != latent_span_len:
        raise ValueError(
            f"expected {latent_span_len} <|latent_pad|> tokens, tokenizer produced {n_latent} -- "
            f"tokenizer likely merged/split the repeated placeholder text."
        )
    return full_inputs, latent_mask


@torch.no_grad()
def evaluate(vlm, val_stream, processor, tokenizer, vision_tower, device, compute_dtype, args):
    """Read-only held-out evaluation (added 2026-09-12 -- see module
    docstring, "Held-out validation"). Runs up to --eval_max_examples
    examples from val_stream through the exact same forward pass and loss
    computation as training (same build_training_example/
    compute_future_embedding), but under torch.no_grad() and with the model
    in eval() mode -- no backward pass, no optimizer step, no effect on
    training whatsoever. Returns (mean_loss_ce, mean_loss_latent, n_examples
    actually evaluated). Restores vlm.train() before returning so the
    caller's training loop resumes exactly as it left off."""
    vlm.eval()
    ce_losses, latent_losses = [], []
    n = min(args.eval_max_examples, len(val_stream))
    for _ in range(n):
        example = val_stream.next_usable_example()
        target = compute_future_embedding(
            example["future_frames"], processor, vision_tower, device, compute_dtype
        )
        inputs, latent_mask = build_training_example(
            example, processor, tokenizer, args.latent_span_len, args.max_answer_chars
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = vlm(**inputs, output_hidden_states=True, return_dict=True)
        h_latent = out.hidden_states[-1][0, latent_mask, :]
        loss_latent = F.mse_loss(h_latent.float(), target.unsqueeze(0).expand_as(h_latent).float())
        ce_losses.append(out.loss.item())
        latent_losses.append(loss_latent.item())
    vlm.train()
    if not ce_losses:
        return None, None, 0
    return sum(ce_losses) / len(ce_losses), sum(latent_losses) / len(latent_losses), n


def save_checkpoint(output_dir, step, vlm, tokenizer, optimizer, scheduler):
    ckpt_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # If vlm is a peft PeftModel (the --use_lora default), this writes only the
    # adapter weights + the fully-trainable embed_tokens/lm_head copies -- NOT a
    # full copy of the 3B-parameter base model. See this file's module docstring,
    # "LoRA fine-tuning" section, for what that means for consuming this checkpoint
    # later (load base weights + peft.PeftModel.from_pretrained, not a plain
    # from_pretrained on this directory alone).
    vlm.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)  # MUST travel with the model -- vocab was resized
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        os.path.join(ckpt_dir, "training_state.pt"),
    )
    log(f"  checkpoint saved: {ckpt_dir}")


def load_model_and_tokenizer(args, device, compute_dtype):
    """Loads Qwen2.5-VL-3B-Instruct, adds the three latent-reasoning special
    tokens, and either wraps it in a LoRA adapter (default, --use_lora) or
    leaves it fully trainable (--no_lora, the original behavior). Returns
    (vlm, tokenizer, processor, vision_tower) with vision_tower guaranteed
    frozen either way."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.qwen_dir)
    tokenizer = processor.tokenizer

    if args.use_lora:
        # LoRA path: the frozen BASE weights never change across training, so there's
        # no reason to reload them from a checkpoint dir -- always load fresh from
        # --qwen_dir, and attach the (small) saved adapter separately below if resuming.
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.qwen_dir, dtype=compute_dtype, device_map=None, low_cpu_mem_usage=True,
        ).to(device)
    else:
        # Full fine-tuning path (original behavior, unchanged): ALL weights change,
        # so resuming means reloading the full checkpoint's weights.
        load_dir = args.resume_from or args.qwen_dir
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            load_dir, dtype=compute_dtype, device_map=None, low_cpu_mem_usage=True,
        ).to(device)

    assert vlm.config.text_config.hidden_size == QWEN_HIDDEN_SIZE, (
        f"Expected hidden_size={QWEN_HIDDEN_SIZE}, got {vlm.config.text_config.hidden_size}"
    )

    if not args.use_lora and args.resume_from:
        # Full-FT resume: tokenizer must come from the checkpoint too (it carries the
        # resized vocab -- add_special_tokens below would double-add otherwise).
        tokenizer = type(tokenizer).from_pretrained(args.resume_from)
        processor.tokenizer = tokenizer
        log(f"resumed tokenizer + full weights from {args.resume_from}")
    else:
        # Fresh full-FT run, OR any LoRA run (fresh or resumed): always start from the
        # clean base tokenizer and add the same 3 tokens the same way, so vocab/ids
        # come out identical every time regardless of whether this is a resume.
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [LATENT_START, LATENT_END, LATENT_PAD]}
        )
        if num_added > 0:
            # mean_resizing=True (explicit, 2026-09-24) -- see module docstring, "New-token
            # embedding initialization". Draws each new row from a distribution fit to the
            # EXISTING embedding matrix's own mean/covariance instead of small independent
            # random noise -- this is already transformers' own default (since ~4.46), but
            # pinned explicitly here so behavior doesn't silently change if a future
            # transformers release flips that default (huggingface/transformers#35357
            # proposes exactly that, for performance reasons on large-vocab models).
            vlm.resize_token_embeddings(len(tokenizer), mean_resizing=True)
            log_new_token_embedding_norms(vlm, tokenizer, [LATENT_START, LATENT_END, LATENT_PAD])
        log(f"added {num_added} latent-reasoning special tokens; vocab size now {len(tokenizer)}")

    vision_tower = get_vision_tower(vlm)
    vision_tower.requires_grad_(False)  # frozen: only used to compute the target, never trained here

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, PeftModel

        if args.resume_from:
            vlm = PeftModel.from_pretrained(vlm, args.resume_from, is_trainable=True)
            tokenizer = type(tokenizer).from_pretrained(args.resume_from)
            processor.tokenizer = tokenizer
            log(f"resumed LoRA adapter (+ embed_tokens/lm_head) from {args.resume_from}")
        else:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=args.lora_target_modules.split(","),
                # The 3 latent-reasoning tokens are brand new vocab rows with no
                # pretrained weight for a low-rank delta to adapt -- they need real
                # gradient updates from scratch, so keep these two layers fully
                # trainable (peft copies + unfreezes them; the base copy stays frozen).
                modules_to_save=["embed_tokens", "lm_head"],
            )
            vlm = get_peft_model(vlm, lora_config)
            log("applied a fresh LoRA config (embed_tokens/lm_head kept fully trainable "
                "for the new latent tokens; everything else outside target_modules stays frozen)")

        # Defensive: target_modules is matched by leaf module NAME anywhere in the
        # model. This hasn't been confirmed against Qwen2.5-VL's actual vision-tower
        # module names -- if its ViT happens to reuse a name like "q_proj" for its own
        # attention, peft could silently attach trainable LoRA adapters there too,
        # which would break the "vision tower is entirely frozen, used only to compute
        # an immutable loss target" design this script depends on. Re-freeze anything
        # under the vision tower's parameter names regardless of what LoRA just did.
        n_refrozen = 0
        for name, param in vlm.named_parameters():
            if "visual" in name and param.requires_grad:
                param.requires_grad_(False)
                n_refrozen += 1
        if n_refrozen:
            log(f"  re-froze {n_refrozen} vision-tower parameter(s) LoRA had made trainable "
                f"by a module-name collision -- see comment above")
        vlm.print_trainable_parameters()

    return vlm, tokenizer, processor, vision_tower


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--metadata_dir", required=True, help="Path to the staged TwiFF-2.7M slice (save_to_disk dir)")
    p.add_argument("--video_root", required=True, help="Directory of Panda-70M clips named to match TwiFF's `video` column")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", default=None, help="A previous step_NNNN checkpoint dir to resume from")
    p.add_argument("--exclude_flagged_json", default=None,
                   help="Path to screen_panda70m_clip_quality.py's quality_screen_flagged.json. Every "
                        "TwiFF row whose `video` filename appears in this file (i.e. was judged "
                        "MISMATCH, PARSE_ERROR, or NO_METADATA) is excluded from this run's training "
                        "stream entirely -- see module docstring, 'Quality-screen filtering'.")

    p.add_argument("--val_metadata_dir", default=None,
                   help="Path to the staged TwiFF-2.7M OFFICIAL VALIDATION slice (save_to_disk dir, "
                        "produced by data_staging.sh's validation-split block) -- a disjoint set of "
                        "rows/source videos from --metadata_dir, not a resplit of it. If set, turns on "
                        "periodic held-out evaluation -- see module docstring, 'Held-out validation'. "
                        "If unset (default), this run has no held-out signal at all: train loss alone "
                        "cannot distinguish learning from memorization on a dataset this small.")
    p.add_argument("--val_video_root", default=None,
                   help="Directory of Panda-70M clips for the validation split. Defaults to --video_root "
                        "if unset, but should normally be a SEPARATE directory (see "
                        "download_panda70m_clips.sh's validation-set comment block) -- validation clips "
                        "must never live alongside training clips.")
    p.add_argument("--val_exclude_flagged_json", default=None,
                   help="Same idea as --exclude_flagged_json, applied to the validation stream, from "
                        "the validation clips' own screen_panda70m_clip_quality.py run.")
    p.add_argument("--eval_every", type=int, default=None,
                   help="Run held-out evaluation every this many steps. Defaults to --save_every if "
                        "unset (i.e. eval piggybacks on the checkpoint cadence unless told otherwise). "
                        "Has no effect if --val_metadata_dir is not set.")
    p.add_argument("--eval_max_examples", type=int, default=200,
                   help="Cap on how many validation examples to run per evaluation call -- keeps eval "
                        "cost bounded regardless of how large the validation split is (currently 1,451 "
                        "TwiFF rows). Each eval example costs roughly one forward pass (no backward), "
                        "so 200 examples is a real but bounded addition to wall-clock time, not a full "
                        "pass over validation every time.")

    p.add_argument("--frames_root", default=None,
                   help="Directory of pre-extracted frame JPEGs written by extract_stage0_frames.py "
                        "(<frames_root>/<clip_stem>/<frame_idx>.jpg), for the TRAINING stream. When "
                        "set, TwiFFStage0Stream reads frames from here instead of live-decoding the "
                        "raw clip via decord on every draw -- see module docstring, 'Pre-extracted "
                        "frame cache'. When unset (default), behavior is unchanged from before this "
                        "flag existed: raw clips are live-decoded from --video_root exactly as always. "
                        "--video_root is still required either way (kept as the canonical clip-identity "
                        "reference and as the live-decode fallback for any clip not yet extracted).")
    p.add_argument("--val_frames_root", default=None,
                   help="Same idea as --frames_root, applied to the VALIDATION stream (--val_video_root "
                        "/ --val_metadata_dir). Independent of --frames_root -- one stream can read "
                        "from the extracted-frame cache while the other still live-decodes, e.g. if "
                        "only train or only val has been extracted so far.")

    p.add_argument("--wandb_project", default=None,
                   help="W&B project name. Unset (default) = W&B logging is entirely off, and wandb is "
                        "never imported. Set this to turn it on -- see module docstring, 'Weights & "
                        "Biases logging'. Requires `wandb login` to have already been run once on this "
                        "cluster (or WANDB_API_KEY set in the job's environment) -- no API key CLI flag "
                        "on purpose.")
    p.add_argument("--wandb_entity", default=None, help="W&B entity/team (optional -- defaults to your account default)")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (default: 'stage0-<SLURM job id or timestamp>')")

    p.add_argument("--latent_span_len", type=int, default=4, help="Future-L1's optimal L_max for text answers")
    p.add_argument("--lambda_latent", type=float, default=0.1, help="Weight on L_latent (Future-L1's value)")
    p.add_argument("--max_answer_chars", type=int, default=1500, help="Safety truncation on TwiFF answer text")

    p.add_argument("--use_lora", action="store_true", default=True,
                   help="Fine-tune via LoRA adapters instead of updating all 3B parameters (default: on). "
                        "See this file's module docstring, 'LoRA fine-tuning' section, for why this is the "
                        "default given how small the current Panda-70M pilot dataset is.")
    p.add_argument("--no_lora", dest="use_lora", action="store_false",
                   help="Fully fine-tune all of Qwen2.5-VL-3B's parameters instead (the original behavior "
                        "before LoRA support was added) -- lower --lr back down (e.g. 1e-5) if you use this.")
    p.add_argument("--lora_r", type=int, default=8, help="LoRA rank -- kept small/conservative given the "
                   "current tiny pilot dataset (see module docstring); raise once real data volume exists")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA scaling factor, conventionally ~2x r")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated Linear module names inside the LANGUAGE MODEL backbone to wrap in "
                        "LoRA adapters. Attention projections only by default (conservative for the current "
                        "small dataset) -- add gate_proj,up_proj,down_proj for the MLP layers too once a "
                        "real-scale Panda-70M download exists. The vision tower is excluded and explicitly "
                        "re-frozen regardless of what matches this list -- see load_model_and_tokenizer().")

    p.add_argument("--batch_size", type=int, default=1, help="Real micro-batch size; kept at 1 by default -- "
                    "even with LoRA's much lower memory footprint than full fine-tuning, a single VLM "
                    "example (video frames + long context) still isn't cheap")
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4, help="Default tuned for LoRA (freshly-initialized "
                   "low-rank adapters typically want a much higher LR than full fine-tuning) -- if you pass "
                   "--no_lora, lower this back down to roughly 1e-5 or it will likely be too aggressive for "
                   "updating all 3B weights directly")
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="'auto' falls back to CPU with float32 if no GPU -- useful only for exercising "
                        "this script's own data/skip-handling logic on a login node; real training needs a GPU.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cuda_ok = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_ok:
        log("FATAL: --device cuda requested but CUDA is not available.")
        sys.exit(1)

    # A GPU can be *allocated* by SLURM (--gres=gpu:1) while torch still can't
    # see it -- e.g. torch built for a newer CUDA than this node's driver
    # supports (see claude/latentvans-handoff.md, item 9: hit for real on
    # cn8/cn9 of the `ada` partition). --device auto would otherwise silently
    # fall back to CPU and quietly train a 3B-parameter model there for
    # hours. If SLURM's own env vars show a GPU was handed to this job but
    # torch can't use it, that is never something to fall back from silently.
    gpu_was_allocated = bool(
        os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_GPUS_ON_NODE")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if args.device == "auto" and not cuda_ok and gpu_was_allocated:
        log("FATAL: this job was allocated a GPU (SLURM_JOB_GPUS/SLURM_GPUS_ON_NODE/"
            "CUDA_VISIBLE_DEVICES set) but torch.cuda.is_available() is False -- almost certainly "
            "the torch/CUDA driver mismatch documented in claude/latentvans-handoff.md (torch built "
            "for a newer CUDA than this node's driver supports; check with plain `nvidia-smi`, not "
            "--query-gpu). Refusing to silently train a 3B-parameter model on CPU. Fix the torch "
            "install (pin to a CUDA build <= this node's driver ceiling) and resubmit, or pass "
            "--device cpu explicitly if you actually want a CPU sanity run.")
        sys.exit(1)

    use_cuda = cuda_ok and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    compute_dtype = dtype_from_str(args.dtype)
    if not use_cuda and compute_dtype != torch.float32:
        log(f"device=cpu: overriding --dtype {args.dtype} -> float32")
        compute_dtype = torch.float32
    if not use_cuda:
        log("=" * 70)
        log("RUNNING ON CPU. Real Stage 0 SFT on a 3B-parameter VLM is not viable on CPU -- this path "
            "exists only for exercising this script's own data/skip-handling logic on a login node "
            "with no GPU allocated at all. If a GPU WAS allocated and you're seeing this, that's the "
            "bug this check above is meant to catch; you shouldn't be able to reach this banner in "
            "that case.")
        log("=" * 70)
    log(f"torch={torch.__version__} device={device} compute_dtype={compute_dtype} use_lora={args.use_lora}")

    # W&B is entirely opt-in (see module docstring, "Weights & Biases logging") -- off
    # unless --wandb_project is set, and its own failure (missing package, no network,
    # not logged in) must never take down a multi-hour training job. wandb_run stays
    # None whenever logging is off OR init failed, and every wandb.log() call below is
    # guarded on it.
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            run_name = args.wandb_run_name or f"stage0-{os.environ.get('SLURM_JOB_ID', time.strftime('%Y%m%d-%H%M%S'))}"
            wandb_run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity, name=run_name,
                config=vars(args), resume="allow",
                id=(f"stage0-{args.resume_from.rstrip('/').split('/')[-1]}" if args.resume_from else None),
            )
            log(f"W&B logging ON: project={args.wandb_project} run={run_name} url={wandb_run.url}")
        except Exception as e:
            log(f"WARNING: --wandb_project was set but W&B init failed ({e}) -- continuing WITHOUT "
                f"W&B logging. metrics.jsonl is unaffected. Check `wandb login` was run on this cluster "
                f"and that this node has network access.")
            wandb_run = None

    # decord is only needed by a stream that's still live-decoding raw video -- if
    # --frames_root covers the training stream AND either validation is off or
    # --val_frames_root covers it too, no stream ever calls load_video_frames() this
    # run, so don't require decord to be importable just to start. See module
    # docstring, "Pre-extracted frame cache".
    needs_decord = (not args.frames_root) or (args.val_metadata_dir and not args.val_frames_root)
    try:
        import transformers
        if needs_decord:
            import decord  # noqa: F401 -- needed by load_video_frames; check it up front, not mid-training
        if args.use_lora:
            import peft  # noqa: F401
    except ImportError as e:
        log(f"FATAL: missing dependency ({e}). Try:\n"
            f"  python -m pip install --upgrade 'transformers>=4.49.0' decord peft")
        sys.exit(1)
    from transformers import get_cosine_schedule_with_warmup

    log("=" * 70)
    log(f"Loading Qwen2.5-VL-3B-Instruct ({'LoRA adapter' if args.use_lora else 'fully trainable'})")
    log("=" * 70)
    vlm, tokenizer, processor, vision_tower = load_model_and_tokenizer(args, device, compute_dtype)
    latent_pad_id = tokenizer.convert_tokens_to_ids(LATENT_PAD)

    # Fail fast on the vision-tower dimension mismatch (see verify_vision_target_dim's
    # own docstring for the real crash this catches) -- cheap (one synthetic image) and
    # runs before the expensive TwiFF/video data loading and the training loop itself.
    verify_vision_target_dim(processor, vision_tower, device, compute_dtype)

    if args.gradient_checkpointing:
        if args.use_lora:
            # Required for gradient checkpointing to actually backprop through a
            # frozen base model + LoRA adapters -- without this, the input embedding
            # layer's output doesn't require grad, which silently breaks the
            # checkpointed backward pass. Not needed (and not called) in the
            # --no_lora path, where every parameter is trainable already.
            vlm.enable_input_require_grads()
        vlm.gradient_checkpointing_enable()
    vlm.train()

    trainable_params = [p for p in vlm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    start_step = 0
    if args.resume_from:
        state_path = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = state["step"]
            log(f"resumed optimizer/scheduler state at step {start_step}")

    log("=" * 70)
    log(f"Loading TwiFF-2.7M slice from {args.metadata_dir}, video_root={args.video_root}")
    log("=" * 70)
    exclude_videos = load_excluded_videos(args.exclude_flagged_json)
    stream = TwiFFStage0Stream(args.metadata_dir, args.video_root, seed=args.seed, exclude_videos=exclude_videos,
                                frames_root=args.frames_root)
    log(f"{len(stream)} row(s) usable for this run (after exclusions, if any)")
    log(f"training stream reads frames from: "
        + (f"pre-extracted cache at {args.frames_root}" if args.frames_root
           else f"live decord decode of {args.video_root}"))

    # Held-out validation (added 2026-09-12) -- see module docstring, "Held-out
    # validation". Entirely opt-in: if --val_metadata_dir isn't set, val_stream stays
    # None and the training loop below simply never calls evaluate(). This is the
    # only thing in this script that can tell memorization apart from generalization
    # on a dataset this small -- train loss alone cannot.
    val_stream = None
    eval_every = args.eval_every or args.save_every
    if args.val_metadata_dir:
        val_video_root = args.val_video_root or args.video_root
        val_exclude_videos = load_excluded_videos(args.val_exclude_flagged_json)
        val_stream = TwiFFStage0Stream(
            args.val_metadata_dir, val_video_root, seed=args.seed, exclude_videos=val_exclude_videos,
            frames_root=args.val_frames_root
        )
        log(f"validation stream: {len(val_stream)} row(s) usable, video_root={val_video_root} "
            f"-- evaluating up to {args.eval_max_examples} example(s) every {eval_every} step(s)")
        log(f"validation stream reads frames from: "
            + (f"pre-extracted cache at {args.val_frames_root}" if args.val_frames_root
               else f"live decord decode of {val_video_root}"))
    else:
        log("WARNING: --val_metadata_dir not set -- this run has NO held-out evaluation. Train loss "
            "alone cannot distinguish real latent-grounding learning from memorization on a dataset "
            "this small -- see module docstring, 'Held-out validation', and "
            "claude/latentvans-handoff.md.")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    try:
        for step in range(start_step, args.max_steps):
            step_t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            ce_losses, latent_losses = [], []

            for _ in range(args.grad_accum_steps):
                example = stream.next_usable_example()
                target = compute_future_embedding(
                    example["future_frames"], processor, vision_tower, device, compute_dtype
                )
                inputs, latent_mask = build_training_example(
                    example, processor, tokenizer, args.latent_span_len, args.max_answer_chars
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                out = vlm(**inputs, output_hidden_states=True, return_dict=True)
                loss_ce = out.loss
                h_latent = out.hidden_states[-1][0, latent_mask, :]  # (latent_span_len, 2048)
                loss_latent = F.mse_loss(h_latent.float(), target.unsqueeze(0).expand_as(h_latent).float())
                loss = (loss_ce + args.lambda_latent * loss_latent) / args.grad_accum_steps
                loss.backward()

                ce_losses.append(loss_ce.item())
                latent_losses.append(loss_latent.item())

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                mean_ce = sum(ce_losses) / len(ce_losses)
                mean_lat = sum(latent_losses) / len(latent_losses)
                lr_now = scheduler.get_last_lr()[0]
                dt = time.time() - step_t0
                log(f"step {step}/{args.max_steps} loss_ce={mean_ce:.4f} loss_latent={mean_lat:.4f} "
                    f"grad_norm={grad_norm.item():.3f} lr={lr_now:.2e} skipped_total={stream.n_skipped_total} "
                    f"({dt:.1f}s/step)")
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({
                        "step": step, "loss_ce": mean_ce, "loss_latent": mean_lat,
                        "grad_norm": grad_norm.item(), "lr": lr_now,
                        "skipped_total": stream.n_skipped_total, "time": time.time(),
                    }) + "\n")
                if wandb_run is not None:
                    try:
                        wandb_run.log({
                            "loss_ce": mean_ce, "loss_latent": mean_lat,
                            "loss_total": mean_ce + args.lambda_latent * mean_lat,
                            "grad_norm": grad_norm.item(), "lr": lr_now,
                            "skipped_total": stream.n_skipped_total, "seconds_per_step": dt,
                        }, step=step)
                    except Exception as e:
                        # A flaky network mid-run must never take down training -- log once,
                        # then stop trying every step so a dead connection doesn't spam the
                        # job log for the rest of a multi-hour run.
                        log(f"WARNING: wandb.log() failed ({e}) -- disabling W&B logging for the "
                            f"rest of this run. metrics.jsonl keeps being written as normal.")
                        wandb_run = None

            # Held-out evaluation (2026-09-12) -- read-only, see evaluate()'s own
            # docstring. Runs on the same cadence as checkpointing by default so a
            # saved checkpoint always has a matching val-loss data point next to it,
            # but --eval_every can decouple the two. step>0 mirrors the checkpoint
            # condition below so step 0 (model barely initialized) isn't wastefully
            # evaluated.
            if val_stream is not None and step > 0 and step % eval_every == 0:
                eval_t0 = time.time()
                val_ce, val_lat, n_eval = evaluate(
                    vlm, val_stream, processor, tokenizer, vision_tower, device, compute_dtype, args
                )
                if n_eval == 0:
                    log(f"  [eval] step {step}: validation stream produced 0 usable examples -- "
                        f"skipping this eval (check --val_video_root has real clips)")
                else:
                    val_total = val_ce + args.lambda_latent * val_lat
                    log(f"  [eval] step {step}: val_loss_ce={val_ce:.4f} val_loss_latent={val_lat:.4f} "
                        f"val_loss_total={val_total:.4f} (n={n_eval}, {time.time() - eval_t0:.1f}s)")
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps({
                            "step": step, "val_loss_ce": val_ce, "val_loss_latent": val_lat,
                            "val_loss_total": val_total, "val_n": n_eval, "time": time.time(),
                        }) + "\n")
                    if wandb_run is not None:
                        try:
                            wandb_run.log({
                                "val_loss_ce": val_ce, "val_loss_latent": val_lat,
                                "val_loss_total": val_total,
                            }, step=step)
                        except Exception as e:
                            log(f"WARNING: wandb.log() (val) failed ({e}) -- disabling W&B logging "
                                f"for the rest of this run. metrics.jsonl keeps being written as normal.")
                            wandb_run = None

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(args.output_dir, step, vlm, tokenizer, optimizer, scheduler)

        save_checkpoint(args.output_dir, args.max_steps, vlm, tokenizer, optimizer, scheduler)
        log("=" * 70)
        log("STAGE 0 TRAINING COMPLETE")
        log("=" * 70)
        if wandb_run is not None:
            wandb_run.finish()

    except NoUsableExamplesError as e:
        log(f"FATAL: {e}")
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        log("FATAL: CUDA OutOfMemoryError. LoRA (the default) already uses far less memory than full "
            "fine-tuning, so if this still happens: try a smaller --batch_size (already 1?), fewer "
            "--grad_accum_steps won't help VRAM (only step frequency), reduce context/future frame "
            "count in the data, or run on a bigger-memory GPU partition. Lowering --lora_r does NOT "
            "meaningfully reduce memory -- activation memory from the forward/backward pass dominates, "
            "not adapter parameter count.")
        sys.exit(1)
    except Exception:
        log("FATAL: unhandled exception.")
        traceback.print_exc()
        sys.exit(1)
