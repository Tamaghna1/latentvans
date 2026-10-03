"""Is a Stage 0 checkpoint grounding each clip, or just predicting the average target?

val_loss_latent alone can't answer this: an MSE of 0.227 is only meaningful
relative to what a trivial predictor gets on the same targets. This script
loads one LoRA checkpoint (same loading path as
evaluate_stage0_latent_grounding.py), runs held-out validation examples
through the exact training forward pass, and compares the model's latent-token
MSE against:

  const_train   predict the mean target of N training clips for every clip
  const_val     predict the mean of the val targets themselves (oracle constant)
  model_const   predict the model's own average output for every clip
  ctx_copy      predict the pooled embedding of the CONTEXT frames (i.e. "the
                future looks like the present")

plus a retrieval test: for each val clip, rank all val targets by distance to
the model's mean latent state; a model that only learned the average scores at
chance, one that grounds per clip ranks its own target near the top.

Targets follow the checkpoint's stage0_target_config.json (Stage 0 v2: layout,
delta, z-score stats, latent head); checkpoints without one are scored with v1's
pooled raw target. For delta targets ctx_copy is skipped (the context is
already subtracted out). For multi-slot targets every comparison uses the
slot-flattened vectors.

Two "is the VLM doing more than recognising the scene?" checks (added 2026-10-03):

  ridge_ctx       a ridge regression from the raw context-frame features (same layout)
                  to the target, fit on the n_train training clips and scored on the
                  same val clips as the model. If the model only matches this, it has
                  learned nothing the context frame alone doesn't give.
  shuffled_q      the model re-run on every val clip with ANOTHER clip's question
                  (same frames). If MSE does not rise, the latent ignores the question.

Writes <output_json> and prints a verdict. Forward passes only, no training.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage0_targets import compute_target, expand_to_slots, frames_features, load_target_stats  # noqa: E402
from wandb_util import flatten, log_summary  # noqa: E402
from train_stage0_latent_grounding import (  # noqa: E402
    QWEN_HIDDEN_SIZE,
    TwiFFStage0Stream,
    build_training_example,
    dtype_from_str,
    load_model_and_tokenizer,
    log,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="A step_NNNN LoRA checkpoint dir")
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--train_metadata_dir", required=True)
    p.add_argument("--train_video_root", required=True)
    p.add_argument("--train_frames_root", default=None)
    p.add_argument("--val_metadata_dir", required=True)
    p.add_argument("--val_video_root", required=True)
    p.add_argument("--val_frames_root", default=None)
    p.add_argument("--n_train", type=int, default=500, help="Training clips used to estimate the train-mean target")
    p.add_argument("--n_val", type=int, default=300)
    p.add_argument("--latent_span_len", type=int, default=4)
    p.add_argument("--max_answer_chars", type=int, default=1500)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_json", required=True)
    p.add_argument("--wandb_project", default=None, help="Log the result to this W&B project (optional)")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def retrieval(queries, targets):
    """queries/targets: (n, d). Rank of the matching target for each query by MSE distance."""
    d = torch.cdist(queries, targets).pow(2) / queries.shape[1]
    ranks = (d < d.diag().unsqueeze(1)).sum(dim=1)  # 0 = matching target is the closest
    n = len(ranks)
    return {
        "top1": (ranks == 0).float().mean().item(),
        "top5": (ranks < 5).float().mean().item(),
        "mean_rank": ranks.float().mean().item() + 1,
        "chance_top1": 1.0 / n,
        "chance_mean_rank": (n + 1) / 2,
    }


def centered_cosine(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    sim = torch.nn.functional.normalize(a, dim=1) @ torch.nn.functional.normalize(b, dim=1).T
    n = len(sim)
    matched = sim.diag().mean().item()
    mismatched = (sim.sum() - sim.diag().sum()).item() / (n * n - n)
    return matched, mismatched


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda")
    compute_dtype = dtype_from_str(args.dtype)

    load_args = argparse.Namespace(use_lora=True, resume_from=args.checkpoint, qwen_dir=args.qwen_dir)
    vlm, tokenizer, processor, vision_tower = load_model_and_tokenizer(load_args, device, compute_dtype)
    vlm.eval()

    cfg = {"target_layout": "pooled", "target_delta": False, "target_stats": None,
           "latent_head": "none", "ce_on_latent_pads": True}
    cfg_path = os.path.join(args.checkpoint, "stage0_target_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg.update(json.load(f))
    log(f"target config: {cfg}")
    stats = load_target_stats(cfg["target_stats"])
    head = None
    if cfg["latent_head"] == "linear":
        head = torch.nn.Linear(QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE).to(device)
        head.load_state_dict(torch.load(os.path.join(args.checkpoint, "latent_head.pt"), map_location=device))
        head.eval()

    def target_of(ex):
        t = compute_target(ex, cfg["target_layout"], cfg["target_delta"], stats, processor, vision_tower,
                           device, compute_dtype)
        return expand_to_slots(t, args.latent_span_len).cpu()

    def context_as_target(ex):
        """The context frame pushed through the same layout/normalization (ctx_copy baseline)."""
        ctx_ex = {"future_frames": ex["context_frames"], "context_frames": ex["context_frames"]}
        t = compute_target(ctx_ex, cfg["target_layout"], False, stats, processor, vision_tower,
                           device, compute_dtype)
        return expand_to_slots(t, args.latent_span_len).cpu()

    log(f"computing {args.n_train} training targets from {args.train_metadata_dir}")
    train_stream = TwiFFStage0Stream(args.train_metadata_dir, args.train_video_root, seed=args.seed,
                                     frames_root=args.train_frames_root)
    def ctx_features(ex):
        """Raw context-frame features in the target's layout, flattened (ridge input)."""
        return frames_features(ex["context_frames"], cfg["target_layout"], processor, vision_tower,
                               device, compute_dtype).flatten().cpu()

    train_targets, train_ctx = [], []
    for i in range(args.n_train):
        ex = train_stream.next_usable_example()
        train_targets.append(target_of(ex))
        train_ctx.append(ctx_features(ex))
        if (i + 1) % 100 == 0:
            log(f"  train targets: {i + 1}/{args.n_train}")
    train_mean = torch.stack(train_targets).mean(0)

    log(f"running {args.n_val} validation examples through {args.checkpoint}")
    val_stream = TwiFFStage0Stream(args.val_metadata_dir, args.val_video_root, seed=args.seed,
                                   frames_root=args.val_frames_root)
    n_val = min(args.n_val, len(val_stream))
    T, C, H, ce, H_shuf, val_ctx = [], [], [], [], [], []
    prev_question = None
    for i in range(n_val):
        ex = val_stream.next_usable_example()
        val_ctx.append(ctx_features(ex))
        T.append(target_of(ex))
        if not cfg["target_delta"]:
            C.append(context_as_target(ex))
        inputs, latent_mask = build_training_example(ex, processor, tokenizer, args.latent_span_len,
                                                     args.max_answer_chars,
                                                     ce_on_latent_pads=cfg["ce_on_latent_pads"])
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = vlm(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][0, latent_mask, :].float()
        H.append((head(h) if head is not None else h).cpu())
        ce.append(out.loss.item())
        # Same frames, previous clip's question (the first clip borrows the last one's below).
        if prev_question is not None:
            shuf_ex = {**ex, "question": prev_question}
            s_inputs, s_mask = build_training_example(shuf_ex, processor, tokenizer, args.latent_span_len,
                                                      args.max_answer_chars,
                                                      ce_on_latent_pads=cfg["ce_on_latent_pads"])
            s_inputs = {k: v.to(device) for k, v in s_inputs.items()}
            hs = vlm(**s_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][0, s_mask, :].float()
            H_shuf.append((head(hs) if head is not None else hs).cpu())
        prev_question = ex["question"]
        if (i + 1) % 50 == 0:
            log(f"  val: {i + 1}/{n_val}")

    T3 = torch.stack(T)           # (n, span, d) per-slot targets
    H = torch.stack(H)            # (n, span, d) latent predictions (head applied if any)
    n_ex = len(T3)

    def mse(pred, target):  # identical reduction to the training loss, averaged over examples
        return (pred - target).pow(2).mean().item()

    model_mse = mse(H, T3)
    # shuffled-question MSE over clips 1..n-1 (clip 0 had no previous question), vs the model on the same clips
    shuffled_q_mse = mse(torch.stack(H_shuf), T3[1:]) if H_shuf else float("nan")
    model_mse_same_clips = mse(H[1:], T3[1:]) if H_shuf else float("nan")

    def ridge_mse():
        x_tr, x_va = torch.stack(train_ctx).double(), torch.stack(val_ctx).double()
        y_tr = torch.stack(train_targets).flatten(1).double()
        y_va = T3.flatten(1).double()
        mu_x, mu_y = x_tr.mean(0), y_tr.mean(0)
        xt, xv = x_tr - mu_x, x_va - mu_x
        k = xt @ xt.T
        best = None
        for lam in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5):
            alpha = torch.linalg.solve(k + lam * torch.eye(len(k), dtype=k.dtype), y_tr - mu_y)
            pred = xv @ xt.T @ alpha + mu_y
            # note: lambda picked on the val clips themselves, so this baseline is if anything optimistic
            m = (pred - y_va).pow(2).mean().item()
            if best is None or m < best[1]:
                best = (lam, m)
        return best
    baselines = {
        "const_train": mse(train_mean.expand(n_ex, -1, -1), T3),
        "const_val": mse(T3.mean(0).expand(n_ex, -1, -1), T3),
        "model_const": mse(H.mean(0).expand(n_ex, -1, -1), T3),
    }
    if C:
        C3 = torch.stack(C)
        baselines["ctx_copy"] = mse(C3, T3)
    ridge_lam, baselines["ridge_ctx"] = ridge_mse()
    r2_vs_const_val = 1 - model_mse / baselines["const_val"]
    T, Hm = T3.flatten(1), H.flatten(1)  # slot-flattened vectors for cosine/retrieval
    matched, mismatched = centered_cosine(Hm, T)
    if C:
        C = C3.flatten(1)
        ctx_matched, ctx_mismatched = centered_cosine(C, T)

    result = {
        "checkpoint": args.checkpoint,
        "n_train_for_mean": len(train_targets),
        "n_val": len(T),
        "val_loss_ce": sum(ce) / len(ce),
        "model_mse": model_mse,
        "baselines_mse": baselines,
        "r2_vs_const_val": r2_vs_const_val,
        "target_spread": T.var(0, unbiased=False).mean().item(),
        "prediction_spread": Hm.var(0, unbiased=False).mean().item(),
        "centered_cosine_model": {"matched": matched, "mismatched": mismatched},
        "retrieval_model": retrieval(Hm, T),
        "r2_ridge_ctx": 1 - baselines["ridge_ctx"] / baselines["const_val"],
        "ridge_ctx_lambda": ridge_lam,
        "shuffled_question": {"model_mse_same_clips": model_mse_same_clips, "shuffled_q_mse": shuffled_q_mse,
                              "relative_increase": shuffled_q_mse / model_mse_same_clips - 1},
        "target_config": cfg,
    }
    if C is not None and len(C):
        result["centered_cosine_ctx"] = {"matched": ctx_matched, "mismatched": ctx_mismatched}
        result["retrieval_ctx_copy"] = retrieval(C, T)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"T": T3, "C": C, "H": H, "train_mean": train_mean},
               os.path.splitext(args.output_json)[0] + "_tensors.pt")

    log(json.dumps(result, indent=2))
    ret = result["retrieval_model"]
    log("=" * 70)
    log(f"model MSE {model_mse:.4f} vs predict-train-mean {baselines['const_train']:.4f} "
        f"vs copy-context {baselines.get('ctx_copy', float('nan')):.4f}  (R^2 vs val mean = {r2_vs_const_val:.3f})")
    sq = result["shuffled_question"]
    log(f"R^2: model {r2_vs_const_val:.3f} vs context-frame ridge {result['r2_ridge_ctx']:.3f}; "
        f"shuffled-question MSE {sq['shuffled_q_mse']:.4f} vs {sq['model_mse_same_clips']:.4f} "
        f"({100 * sq['relative_increase']:+.1f}%)")
    log(f"retrieval top1 {ret['top1']:.3f} (chance {ret['chance_top1']:.3f}), "
        f"mean rank {ret['mean_rank']:.1f} (chance {ret['chance_mean_rank']:.1f})")
    grounded = not (model_mse >= 0.95 * baselines["const_train"] or ret["top1"] < 3 * ret["chance_top1"])
    if not grounded:
        log("VERDICT: close to the average-target baseline -- Stage 0 is NOT grounding individual clips.")
    else:
        log("VERDICT: beats the average-target baseline and retrieves its own target -- per-clip grounding is real.")
    ckpt = os.path.normpath(args.checkpoint)
    log_summary(args.wandb_project,
                args.wandb_run_name or f"diagnose-{os.path.basename(os.path.dirname(ckpt))}-{os.path.basename(ckpt)}",
                {**vars(args), **cfg}, {**flatten(result), "grounded": int(grounded)}, job_type="diagnose")


if __name__ == "__main__":
    main()
