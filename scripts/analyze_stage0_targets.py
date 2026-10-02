"""Measure Stage 0 target variants before training on them, and write their normalization stats.

For n_train training and n_val validation examples (read from the v2 frame
caches), computes context and future features under both layouts in
stage0_targets.py, then for every (layout, delta) target variant reports:

  cos_future_context    mean cosine between future and context features (raw);
                        v1's pooled target measured 0.96 on the old, wrong frames
  ctx_copy_retrieval    rank future targets by distance to the context features;
                        top-1 near 1.0 means the target is mostly "which scene is this"
  zero_mse              val MSE of predicting the train mean (z-space: ~1.0 by construction)
  ctx_copy_mse          val MSE of predicting z(context) (non-delta variants only)
  ridge_r2              R^2 on val of a ridge regression from the context frame's
                        features to the target, fit on train. A lower bound on how
                        much of the target is predictable from what the VLM sees
                        (the VLM also gets the question text). Near 0: the target
                        is unpredictable noise from the context alone. Near 1: the
                        target is a function of the context, i.e. carries little
                        about the future beyond the scene itself.

Writes <output_dir>/target_stats_<layout>_<delta|abs>.pt (mean/std over the
training examples, consumed by train_stage0_latent_grounding.py --target_stats)
and <output_dir>/target_analysis.json. Forward passes through the frozen vision
tower only.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage0_targets import frames_features  # noqa: E402
from train_stage0_latent_grounding import TwiFFStage0Stream, dtype_from_str, get_vision_tower, log  # noqa: E402
from wandb_util import flatten, log_summary  # noqa: E402

LAYOUTS = ("pooled", "quadrants")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--train_metadata_dir", required=True)
    p.add_argument("--train_video_root", required=True)
    p.add_argument("--train_frames_root", required=True)
    p.add_argument("--val_metadata_dir", required=True)
    p.add_argument("--val_video_root", required=True)
    p.add_argument("--val_frames_root", required=True)
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--ridge_lambdas", default="1,10,100,1000,10000")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--wandb_project", default=None, help="Log the variant table to this W&B project (optional)")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


@torch.no_grad()
def collect(stream, n, processor, vision_tower, device, dtype, tag):
    feats = {lay: {"ctx": [], "fut": []} for lay in LAYOUTS}
    for i in range(n):
        ex = stream.next_usable_example()
        for lay in LAYOUTS:
            feats[lay]["ctx"].append(frames_features(ex["context_frames"], lay, processor, vision_tower, device, dtype).cpu())
            feats[lay]["fut"].append(frames_features(ex["future_frames"], lay, processor, vision_tower, device, dtype).cpu())
        if (i + 1) % 250 == 0:
            log(f"  {tag}: {i + 1}/{n} (skipped so far: {stream.n_skipped_total})")
    return {lay: {k: torch.stack(v) for k, v in d.items()} for lay, d in feats.items()}  # (n, S, d)


def retrieval_top1(queries, targets):
    d = torch.cdist(queries, targets)
    return (d.argmin(dim=1) == torch.arange(len(queries))).float().mean().item()


def ridge_fit_predict(x_tr, y_tr, x_va, lam):
    """Kernel (dual) ridge with a linear kernel: works when n_train < feature dim."""
    mu_x, mu_y = x_tr.mean(0), y_tr.mean(0)
    xt, xv, yt = x_tr - mu_x, x_va - mu_x, y_tr - mu_y
    k = xt @ xt.T
    alpha = torch.linalg.solve(k + lam * torch.eye(len(k), dtype=k.dtype), yt)
    return xv @ xt.T @ alpha + mu_y


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    dtype = dtype_from_str(args.dtype)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.qwen_dir)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwen_dir, dtype=dtype).to(device).eval()
    vision_tower = get_vision_tower(vlm)

    train = collect(TwiFFStage0Stream(args.train_metadata_dir, args.train_video_root, seed=args.seed,
                                      frames_root=args.train_frames_root),
                    args.n_train, processor, vision_tower, device, dtype, "train")
    val = collect(TwiFFStage0Stream(args.val_metadata_dir, args.val_video_root, seed=args.seed,
                                    frames_root=args.val_frames_root),
                  args.n_val, processor, vision_tower, device, dtype, "val")
    torch.save({"train": train, "val": val}, os.path.join(args.output_dir, "target_features.pt"))

    lambdas = [float(x) for x in args.ridge_lambdas.split(",")]
    results = {"n_train": args.n_train, "n_val": args.n_val, "variants": {}}
    for lay in LAYOUTS:
        ctx_tr, fut_tr = train[lay]["ctx"].double(), train[lay]["fut"].double()
        ctx_va, fut_va = val[lay]["ctx"].double(), val[lay]["fut"].double()
        cos = torch.nn.functional.cosine_similarity(fut_va.flatten(1), ctx_va.flatten(1), dim=1).mean().item()
        for delta in (False, True):
            y_tr = fut_tr - ctx_tr if delta else fut_tr
            y_va = fut_va - ctx_va if delta else fut_va
            mean, std = y_tr.mean(0), y_tr.std(0).clamp_min(1e-4)
            z_tr, z_va = (y_tr - mean) / std, (y_va - mean) / std
            name = f"{lay}_{'delta' if delta else 'abs'}"
            torch.save({"layout": lay, "delta": delta, "mean": mean.float(), "std": std.float(),
                        "n": len(y_tr)}, os.path.join(args.output_dir, f"target_stats_{name}.pt"))

            zero_mse = z_va.pow(2).mean().item()
            r = {
                "cos_future_context": cos,
                "ctx_copy_retrieval_top1": retrieval_top1(ctx_va.flatten(1), fut_va.flatten(1)),
                "chance_top1": 1.0 / len(fut_va),
                "zero_mse": zero_mse,
                "raw_target_var": y_va.var(0).mean().item(),
            }
            if not delta:
                r["ctx_copy_mse"] = (((ctx_va - mean) / std) - z_va).pow(2).mean().item()
            best = None
            for lam in lambdas:
                pred = ridge_fit_predict(ctx_tr.flatten(1), z_tr.flatten(1), ctx_va.flatten(1), lam)
                mse = (pred - z_va.flatten(1)).pow(2).mean().item()
                if best is None or mse < best[1]:
                    best = (lam, mse)
            r["ridge_best_lambda"], r["ridge_mse"] = best
            r["ridge_r2"] = 1 - best[1] / zero_mse
            results["variants"][name] = r
            log(f"{name:>16}: cos(fut,ctx)={cos:.3f} ctx-copy top1={r['ctx_copy_retrieval_top1']:.3f} "
                f"zero_mse={zero_mse:.3f} ridge_r2={r['ridge_r2']:.3f} (lambda={best[0]:g})"
                + (f" ctx_copy_mse={r['ctx_copy_mse']:.3f}" if "ctx_copy_mse" in r else ""))

    with open(os.path.join(args.output_dir, "target_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(json.dumps(results, indent=2))
    cols = ["variant", "cos_future_context", "ctx_copy_retrieval_top1", "zero_mse", "ctx_copy_mse",
            "ridge_r2", "ridge_best_lambda"]
    rows = [[name] + [r.get(c) for c in cols[1:]] for name, r in results["variants"].items()]
    log_summary(args.wandb_project, args.wandb_run_name, vars(args), flatten(results),
                tables={"target_variants": (cols, rows)}, job_type="analyze-targets")


if __name__ == "__main__":
    main()
