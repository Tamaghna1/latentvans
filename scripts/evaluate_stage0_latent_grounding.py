#!/usr/bin/env python
"""
Stage 0 -- Validation-only evaluation for LatentVANS.

Purpose (added 2026-09-12, at explicit user request: "i want to run just
validation only now and see the graph"): the user has already trained a
Stage 0 checkpoint (train_stage0_latent_grounding.py, LoRA, ~1,130 effective
usable train clips after item-12 decode attrition -- see
claude/latentvans-handoff.md item 13/14) WITHOUT a held-out validation split,
because that split didn't exist in the pipeline yet. Held-out validation
support has since been added, and TwiFF-2.7M's official `validation` split
has (per the conversation this script was written from) been staged and its
clips downloaded/quality-screened. This script does NOT retrain anything --
it re-evaluates the *existing* checkpoints already saved by that completed
training run against the held-out validation stream, so you can produce the
train-loss-vs-val-loss comparison graph retroactively instead of having to
rerun the whole 2000-step training job with --val_metadata_dir wired in from
the start.

What it does
------------
train_stage0_latent_grounding.py saved a checkpoint every --save_every steps
(default 200) under --output_dir/step_<N>/ -- for a 2000-step run with the
defaults, that's step_200, step_400, ..., step_2000 (10 checkpoints). For
each of those checkpoints, in ascending step order, this script:
  1. Loads the frozen Qwen2.5-VL-3B-Instruct base weights fresh from
     --qwen_dir, then attaches that checkpoint's saved LoRA adapter +
     embed_tokens/lm_head via peft.PeftModel.from_pretrained -- the exact
     same loading path train_stage0_latent_grounding.py's --resume_from
     already uses, reused here unmodified (imported, not reimplemented).
  2. Builds a FRESH TwiFFStage0Stream over the validation split with the
     same --seed every time, so every checkpoint is evaluated against the
     identical sequence of validation examples -- an apples-to-apples
     comparison across steps, not ten different random subsamples.
  3. Runs train_stage0_latent_grounding.py's own evaluate() (read-only,
     torch.no_grad(), vlm.eval()) for up to --eval_max_examples validation
     examples, and records val_loss_ce / val_loss_latent / val_loss_total.
  4. Frees the reloaded model before moving to the next checkpoint.

Reloading the full base model per checkpoint is slower than swapping
adapters on one long-lived model, but it's the simplest thing that is
definitely correct (LoRA checkpoints here also carry modules_to_save copies
of embed_tokens/lm_head, which are real parameter swaps, not just adapter
deltas -- not worth the risk of a subtly-stale swapped-in-place model for a
one-off evaluation script). If this needs to run often, revisit.

Output
------
- <output_dir>/eval_metrics.jsonl -- one line per evaluated checkpoint:
  {"step", "val_loss_ce", "val_loss_latent", "val_loss_total", "n_eval"}.
- <output_dir>/val_vs_train_loss.png -- val_loss_total by step, overlaid
  against the ORIGINAL run's train loss_total (recomputed from
  --train_metrics_path's loss_ce/loss_latent fields, loss_ce + lambda *
  loss_latent) on the same step axis -- this is the actual "does val loss
  track train loss down, or plateau/diverge while train loss keeps
  falling" comparison job 57910 was missing. Skipped with a warning (not a
  hard failure) if matplotlib isn't installed -- eval_metrics.jsonl is
  still written either way and can be plotted from anywhere.
- Optionally logs to W&B if --wandb_project is set. Two modes (added
  2026-09-13, at explicit user request "add wandb analysis too"):
    * Default: a fresh run (name "stage0-eval-<job/timestamp>") gets
      val_loss_ce/val_loss_latent/val_loss_total logged per step, plus the
      two plot PNGs as wandb.Image, plus a summary wandb.Table with one row
      per evaluated checkpoint (step, train_loss_total, val_loss_total,
      gap = val - train) -- a fresh run is the safe default because it can
      never corrupt the original training run's history.
    * If --wandb_resume_run_id is also set (the original training run's
      W&B run ID, e.g. "a1b2c3d4" from its run URL
      wandb.ai/<entity>/<project>/runs/<id>): RESUMES that exact run
      instead of creating a new one, so val_loss_* lands as additional
      points ON THE SAME loss_total chart the training run already has --
      the actual apples-to-apples overlay, not two separate dashboards to
      eyeball side by side. Uses wandb.init(id=..., resume="must"), which
      fails loudly (not silently falls back to a new run) if that run ID
      doesn't exist or isn't resumable, since silently landing in the
      wrong place would be worse than just erroring.

Prerequisites
-------------
The validation split's clips must already be downloaded and (optionally)
quality-screened -- i.e. download_panda70m_clips.sh and
screen_panda70m_clip_quality.sh must have been run with
METADATA_DIR/OUTPUT_DIR (or CLIPS_DIR) pointed at the validation slice, per
those scripts' own comment blocks. This script fails fast with a clear
message (via TwiFFStage0Stream/NoUsableExamplesError, reused unmodified) if
--val_video_root has no usable clips yet -- it does not download anything
itself.

Usage
-----
    python evaluate_stage0_latent_grounding.py \\
        --qwen_dir /scratch/users/anirban/tamaghnam/latentvans/checkpoints/Qwen2.5-VL-3B-Instruct \\
        --checkpoint_root /scratch/users/anirban/tamaghnam/latentvans/checkpoints/stage0_run1 \\
        --val_metadata_dir /scratch/users/anirban/tamaghnam/latentvans/data/twiff_metadata_val \\
        --val_video_root /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips_val \\
        --val_exclude_flagged_json /scratch/users/anirban/tamaghnam/latentvans/data/panda70m_clips_val/quality_screen_flagged.json \\
        --eval_max_examples 200

Needs the same deps train_stage0_latent_grounding.py does (transformers,
decord, peft) plus matplotlib for the plot (optional -- see above). Expects
train_stage0_latent_grounding.py to sit alongside this file (same
$SCRATCH/scripts/ directory) -- its functions are imported, not copied.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_stage0_latent_grounding import (  # noqa: E402
    LATENT_PAD,
    QWEN_HIDDEN_SIZE,
    TwiFFStage0Stream,
    dtype_from_str,
    evaluate,
    load_excluded_videos,
    load_model_and_tokenizer,
    log,
    verify_vision_target_dim,
)


STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def discover_checkpoints(checkpoint_root, only_steps=None):
    found = {}
    for path in glob.glob(os.path.join(checkpoint_root, "step_*")):
        name = os.path.basename(path)
        m = STEP_DIR_RE.match(name)
        if m and os.path.isdir(path):
            found[int(m.group(1))] = path
    if not found:
        log(f"FATAL: no step_<N> checkpoint directories found under {checkpoint_root} -- "
            f"is this the --output_dir a train_stage0_latent_grounding.py run actually wrote to?")
        sys.exit(1)
    steps = sorted(found)
    if only_steps is not None:
        missing = [s for s in only_steps if s not in found]
        if missing:
            log(f"FATAL: --steps requested {missing} but no matching step_<N> dir exists under "
                f"{checkpoint_root} (found: {steps})")
            sys.exit(1)
        steps = sorted(only_steps)
    log(f"will evaluate {len(steps)} checkpoint(s): {steps}")
    return [(s, found[s]) for s in steps]


def read_train_loss_curve(train_metrics_path, lambda_latent):
    """Recomputes loss_total = loss_ce + lambda_latent * loss_latent from the
    ORIGINAL training run's metrics.jsonl (train_stage0_latent_grounding.py's
    own --log_every-cadence lines) so it can be overlaid against this
    script's val curve on the same step axis. Returns (steps, totals), both
    empty if the file doesn't exist or has no train rows -- never fatal,
    this is a nice-to-have overlay, not a requirement."""
    if not train_metrics_path or not os.path.exists(train_metrics_path):
        log(f"no train metrics found at {train_metrics_path!r} -- plot will show val loss only")
        return [], []
    steps, totals = [], []
    with open(train_metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "loss_ce" in row and "loss_latent" in row:  # a TRAIN row, not a val_* row
                steps.append(row["step"])
                totals.append(row["loss_ce"] + lambda_latent * row["loss_latent"])
    log(f"loaded {len(steps)} train-loss point(s) from {train_metrics_path} for the overlay")
    return steps, totals


def maybe_plot(output_dir, eval_rows, train_steps, train_totals):
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display on a compute node
        import matplotlib.pyplot as plt
    except ImportError:
        log("WARNING: matplotlib not installed -- skipping val_vs_train_loss.png. "
            "eval_metrics.jsonl was still written; plot it yourself from there "
            "(pip install matplotlib and re-run just the plotting, or use pandas/wandb).")
        return {}

    val_steps = [r["step"] for r in eval_rows]
    val_totals = [r["val_loss_total"] for r in eval_rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    if train_steps:
        ax.plot(train_steps, train_totals, label="train loss_total", color="tab:blue", alpha=0.7)
    ax.plot(val_steps, val_totals, label="val loss_total", color="tab:red", marker="o")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss_ce + lambda * loss_latent")
    ax.set_title("Stage 0: train vs held-out validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path = os.path.join(output_dir, "val_vs_train_loss.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log(f"wrote plot: {out_path}")

    # Second panel: loss_ce and loss_latent split out separately for val, in
    # case the combined total hides which term is actually diverging.
    fig2, (ax_ce, ax_lat) = plt.subplots(1, 2, figsize=(13, 5))
    ax_ce.plot(val_steps, [r["val_loss_ce"] for r in eval_rows], marker="o", color="tab:red")
    ax_ce.set_title("val loss_ce by step")
    ax_ce.set_xlabel("training step")
    ax_ce.grid(alpha=0.3)
    ax_lat.plot(val_steps, [r["val_loss_latent"] for r in eval_rows], marker="o", color="tab:purple")
    ax_lat.set_title("val loss_latent by step")
    ax_lat.set_xlabel("training step")
    ax_lat.grid(alpha=0.3)
    out_path2 = os.path.join(output_dir, "val_loss_components.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    log(f"wrote plot: {out_path2}")

    return {"val_vs_train_loss": out_path, "val_loss_components": out_path2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--checkpoint_root", required=True,
                   help="The --output_dir a train_stage0_latent_grounding.py run wrote its "
                        "step_<N>/ checkpoints into.")
    p.add_argument("--steps", default=None,
                   help="Comma-separated list of specific step numbers to evaluate (must each have "
                        "a step_<N> dir under --checkpoint_root). Default: evaluate every step_<N> "
                        "checkpoint found there.")
    p.add_argument("--val_metadata_dir", required=True)
    p.add_argument("--val_video_root", required=True)
    p.add_argument("--val_exclude_flagged_json", default=None)
    p.add_argument("--eval_max_examples", type=int, default=200,
                   help="Cap per checkpoint. The same --eval_max_examples validation examples "
                        "(identical order, same --seed) are used for every checkpoint evaluated, "
                        "so the numbers are directly comparable across steps.")
    p.add_argument("--latent_span_len", type=int, default=4,
                   help="MUST match the value the original training run used (its default was 4 "
                        "and this script's default matches) -- a mismatch will make "
                        "build_training_example() raise, not silently produce wrong numbers.")
    p.add_argument("--max_answer_chars", type=int, default=1500)
    p.add_argument("--lambda_latent", type=float, default=0.1,
                   help="MUST match the training run's --lambda_latent (default 0.1 both places) -- "
                        "used to compute loss_total for both the val curve and the train overlay.")
    p.add_argument("--output_dir", default=None,
                   help="Where eval_metrics.jsonl and the plots go. Default: "
                        "<checkpoint_root>/eval_only/")
    p.add_argument("--train_metrics_path", default=None,
                   help="Path to the ORIGINAL training run's metrics.jsonl, for the train-loss "
                        "overlay on the plot. Default: <checkpoint_root>/metrics.jsonl (where "
                        "train_stage0_latent_grounding.py already writes it).")

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None,
                   help="Default: 'stage0-eval-<SLURM job id or timestamp>'. Ignored if "
                        "--wandb_resume_run_id is set (the resumed run keeps its own name).")
    p.add_argument("--wandb_resume_run_id", default=None,
                   help="W&B run ID of the ORIGINAL training run to resume instead of creating a new "
                        "eval run -- val_loss_* then lands on the same chart as that run's train "
                        "loss_ce/loss_latent/loss_total, giving a real train-vs-val overlay in one "
                        "W&B dashboard instead of two runs to compare by hand. Find it in the "
                        "training run's URL (wandb.ai/<entity>/<project>/runs/<this-id>) or via "
                        "`wandb.Api().runs(...)`. Fails loudly if the run ID doesn't exist/can't be "
                        "resumed, rather than silently falling back to a fresh run.")

    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=0,
                   help="MUST match the training run's --seed if you want val_metadata_dir row "
                        "ordering to be reproducible run-to-run (doesn't need to match the training "
                        "seed itself -- this only controls the validation stream's own shuffle, a "
                        "completely separate stream from the training one).")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir or os.path.join(args.checkpoint_root, "eval_only")
    os.makedirs(output_dir, exist_ok=True)
    train_metrics_path = args.train_metrics_path or os.path.join(args.checkpoint_root, "metrics.jsonl")

    cuda_ok = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_ok:
        log("FATAL: --device cuda requested but CUDA is not available.")
        sys.exit(1)
    gpu_was_allocated = bool(
        os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_GPUS_ON_NODE")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if args.device == "auto" and not cuda_ok and gpu_was_allocated:
        log("FATAL: this job was allocated a GPU but torch.cuda.is_available() is False -- see "
            "claude/latentvans-handoff.md's torch/CUDA driver notes. Refusing to silently run a "
            "3B-parameter model's forward pass on CPU ten times over.")
        sys.exit(1)
    use_cuda = cuda_ok and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    compute_dtype = dtype_from_str(args.dtype)
    if not use_cuda and compute_dtype != torch.float32:
        log(f"device=cpu: overriding --dtype {args.dtype} -> float32")
        compute_dtype = torch.float32
    log(f"torch={torch.__version__} device={device} compute_dtype={compute_dtype}")

    try:
        import decord  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        log(f"FATAL: missing dependency ({e}). Try:\n"
            f"  python -m pip install --upgrade 'transformers>=4.49.0' decord peft")
        sys.exit(1)

    only_steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    checkpoints = discover_checkpoints(args.checkpoint_root, only_steps)

    if not os.path.isdir(args.val_video_root) or not glob.glob(os.path.join(args.val_video_root, "*.mp4")):
        log(f"WARNING: {args.val_video_root} has no .mp4 files -- every checkpoint below will "
            f"likely report n_eval=0. Run download_panda70m_clips.sh against the validation slice "
            f"first (see that script's own comment block for the METADATA_DIR/OUTPUT_DIR overrides).")

    val_exclude_videos = load_excluded_videos(args.val_exclude_flagged_json)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            if args.wandb_resume_run_id:
                # Resume the ORIGINAL training run so val_loss_* lands on the same chart as its
                # train loss_ce/loss_latent/loss_total -- see module docstring, "add wandb analysis
                # too". resume="must" fails loudly instead of silently creating a fresh run if this
                # ID is wrong, since landing eval numbers in the wrong run would be worse than
                # erroring here.
                wandb_run = wandb.init(
                    project=args.wandb_project, entity=args.wandb_entity,
                    id=args.wandb_resume_run_id, resume="must",
                )
                log(f"W&B logging ON (RESUMING original training run): "
                    f"project={args.wandb_project} run_id={args.wandb_resume_run_id} url={wandb_run.url}")
            else:
                run_name = args.wandb_run_name or f"stage0-eval-{os.environ.get('SLURM_JOB_ID', time.strftime('%Y%m%d-%H%M%S'))}"
                wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args))
                log(f"W&B logging ON (fresh eval run): project={args.wandb_project} run={run_name} url={wandb_run.url}")
        except Exception as e:
            log(f"WARNING: --wandb_project was set but W&B init failed ({e}) -- continuing without it. "
                f"If you passed --wandb_resume_run_id, double-check it against the training run's URL.")
            wandb_run = None

    metrics_path = os.path.join(output_dir, "eval_metrics.jsonl")
    eval_rows = []
    eval_args = argparse.Namespace(
        eval_max_examples=args.eval_max_examples,
        latent_span_len=args.latent_span_len,
        max_answer_chars=args.max_answer_chars,
    )

    for step, ckpt_dir in checkpoints:
        log("=" * 70)
        log(f"Checkpoint step {step}: {ckpt_dir}")
        log("=" * 70)
        load_args = argparse.Namespace(use_lora=True, resume_from=ckpt_dir, qwen_dir=args.qwen_dir)
        vlm, tokenizer, processor, vision_tower = load_model_and_tokenizer(load_args, device, compute_dtype)
        verify_vision_target_dim(processor, vision_tower, device, compute_dtype)

        # Fresh stream every checkpoint, same seed -- every checkpoint sees the identical
        # sequence of validation examples, so the numbers across steps are comparable.
        val_stream = TwiFFStage0Stream(
            args.val_metadata_dir, args.val_video_root, seed=args.seed, exclude_videos=val_exclude_videos
        )
        log(f"validation stream: {len(val_stream)} row(s) usable")

        t0 = time.time()
        val_ce, val_lat, n_eval = evaluate(
            vlm, val_stream, processor, tokenizer, vision_tower, device, compute_dtype, eval_args
        )
        dt = time.time() - t0

        if n_eval == 0:
            log(f"  step {step}: 0 usable validation examples -- skipping (see the warning above "
                f"about --val_video_root)")
        else:
            val_total = val_ce + args.lambda_latent * val_lat
            log(f"  step {step}: val_loss_ce={val_ce:.4f} val_loss_latent={val_lat:.4f} "
                f"val_loss_total={val_total:.4f} (n={n_eval}, {dt:.1f}s)")
            row = {
                "step": step, "val_loss_ce": val_ce, "val_loss_latent": val_lat,
                "val_loss_total": val_total, "n_eval": n_eval, "time": time.time(),
            }
            eval_rows.append(row)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        "val_loss_ce": val_ce, "val_loss_latent": val_lat, "val_loss_total": val_total,
                    }, step=step)
                except Exception as e:
                    log(f"WARNING: wandb.log() failed ({e}) -- disabling W&B logging for the rest "
                        f"of this run. {os.path.basename(metrics_path)} keeps being written as normal.")
                    wandb_run = None

        del vlm
        if use_cuda:
            torch.cuda.empty_cache()

    log("=" * 70)
    log(f"VALIDATION-ONLY EVAL COMPLETE -- {len(eval_rows)}/{len(checkpoints)} checkpoint(s) produced "
        f"usable validation numbers. See {metrics_path}")
    log("=" * 70)

    if eval_rows:
        train_steps, train_totals = read_train_loss_curve(train_metrics_path, args.lambda_latent)
        plot_paths = maybe_plot(output_dir, eval_rows, train_steps, train_totals)

        # W&B analysis, added 2026-09-13 at explicit user request ("add wandb analysis too"): a
        # per-checkpoint train-vs-val-vs-gap table, plus the plot images themselves, so the
        # generalization read lives in W&B too, not just the local PNGs/jsonl. `gap` is the actual
        # memorization signal this whole script exists to produce: gap near 0 and flat/shrinking
        # across steps reads as generalizing; gap growing while train loss keeps falling toward
        # zero reads as memorizing (see claude/latentvans-handoff.md item 13).
        train_by_step = dict(zip(train_steps, train_totals))
        summary_rows = []
        for r in eval_rows:
            train_total = train_by_step.get(r["step"])
            if train_total is None and train_steps:
                nearest = min(train_steps, key=lambda s: abs(s - r["step"]))
                train_total = train_by_step[nearest]
            gap = (r["val_loss_total"] - train_total) if train_total is not None else None
            summary_rows.append({**r, "train_loss_total": train_total, "gap": gap})
        summary_path = os.path.join(output_dir, "train_vs_val_summary.jsonl")
        with open(summary_path, "w") as f:
            for row in summary_rows:
                f.write(json.dumps(row) + "\n")
        log(f"wrote train-vs-val summary (with generalization gap): {summary_path}")

        if wandb_run is not None:
            try:
                import wandb
                table = wandb.Table(columns=["step", "train_loss_total", "val_loss_total", "gap", "n_eval"])
                for row in summary_rows:
                    table.add_data(row["step"], row["train_loss_total"], row["val_loss_total"],
                                    row["gap"], row["n_eval"])
                    if row["gap"] is not None:
                        wandb_run.log({"generalization_gap": row["gap"]}, step=row["step"])
                log_payload = {"train_vs_val_table": table}
                for name, path in (plot_paths or {}).items():
                    if path and os.path.exists(path):
                        log_payload[name] = wandb.Image(path)
                wandb_run.log(log_payload)
                log("logged train_vs_val_table + plot images to W&B")
            except Exception as e:
                log(f"WARNING: failed to log W&B table/images ({e}) -- eval_metrics.jsonl and "
                    f"train_vs_val_summary.jsonl are still the source of truth.")
    else:
        log("WARNING: no checkpoint produced usable validation examples -- nothing to plot. "
            "This almost always means --val_video_root doesn't have the validation clips staged "
            "yet (see PREREQUISITES above).")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
