"""Paired, per-clip comparison of finished VDM-pilot arms.

train_vdm_pilot.py's periodic eval reports one mean over 256 clips, which cannot say
whether a 0.5% gap between arms is real. This reloads each arm's final checkpoint
(Wan LoRA + latent projector), scores EVERY validation clip with exactly the same noise
draws and noise levels across arms, and for each pair of arms reports:

  mean_diff        mean over clips of (loss_a - loss_b); negative = a is better
  ci95             bootstrap 95% interval of that mean (clips resampled, 10k draws)
  frac_a_better    share of clips where arm a has the lower loss
  rel_diff         mean_diff / mean loss of b

A difference whose CI excludes 0 is a real difference on this val set.

Shuffled-conditioning control (added 2026-10-06): every non-null arm is also scored with
the conditioning of a DIFFERENT clip (clip j gets clip j+1's caption / states / latents;
same video, same noise), reported as "<arm>:shuffled". A learned projector can lower the
loss with content-independent "prompt" tokens that adapt Wan to this setting; only the
gap between an arm and its shuffled version measures per-clip information. Per-clip losses
are saved to <output_dir>/per_clip_losses.pt for later slicing (e.g. by question type).
"""
import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vdm_pilot import EVAL_SIGMAS, LatentProjector, WAN_TEXT_DIM, build_cond, flow_loss, load_split, log  # noqa: E402
from wandb_util import log_summary  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm_dirs", nargs="+", required=True, help="final step_N dirs of each arm")
    p.add_argument("--wan_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_val", type=int, default=None, help="default: all val clips")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--wandb_project", default=None)
    return p.parse_args()


def arm_name(cfg):
    if cfg["cond"] == "qwen":
        name = f"qwen:{cfg.get('qwen_layer', 'last')}"
    elif cfg["cond"] in ("latent", "both"):
        name = f"{cfg['cond']}:{os.path.basename(cfg['vlm_dir'].rstrip('/'))}"
    else:
        name = cfg["cond"]
    if cfg.get("text_tag", "caption") != "caption" and cfg["cond"] in ("caption", "both"):
        name += f":text-{cfg['text_tag']}"
    return name + (f":seed{cfg['seed']}" if cfg.get("seed", 0) else "")


@torch.no_grad()
def score_arm(step_dir, wan_dir, batch_size, n_val, base_model, device):
    from peft import LoraConfig, set_peft_model_state_dict

    cfg = json.load(open(os.path.join(step_dir, "config.json")))
    args = argparse.Namespace(**cfg)
    val = load_split(args, "val")
    null_emb = val["null"].to(device).float()
    model = base_model
    if "default" in getattr(model, "peft_config", {}):
        model.delete_adapters("default")
    model.add_adapter(LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                 target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    set_peft_model_state_dict(model, torch.load(os.path.join(step_dir, "wan_lora.pt"), map_location="cpu"))
    model.to(device).eval()
    projector = None
    if cfg["cond"] in ("latent", "both", "qwen"):
        projector = LatentProjector(val["latent"].shape[-1], WAN_TEXT_DIM, 1.0).to(device)
        projector.load_state_dict(torch.load(os.path.join(step_dir, "latent_projector.pt"), map_location=device))
        projector.eval()
    patch = tuple(model.config.patch_size)
    n = min(n_val or len(val["keys"]), len(val["keys"]))
    losses = torch.zeros(n, len(EVAL_SIGMAS))
    shuffled = torch.zeros(n, len(EVAL_SIGMAS)) if cfg["cond"] != "null" else None
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        x0 = val["video"][idx].to(device).float()
        cond = build_cond(args, val, idx, projector, null_emb, device)
        cond_shuf = (build_cond(args, val, [(j + 1) % n for j in idx], projector, null_emb, device)
                     if shuffled is not None else None)
        for si, s in enumerate(EVAL_SIGMAS):
            gens = [torch.Generator(device="cpu").manual_seed(10_000 * j + int(s * 100)) for j in idx]
            noise = torch.stack([torch.randn(x0.shape[1:], generator=g) for g in gens]).to(device)
            sigma = torch.full((len(idx),), s, device=device)
            losses[idx, si] = flow_loss(model, x0, sigma, noise, cond, patch).cpu()
            if cond_shuf is not None:
                shuffled[idx, si] = flow_loss(model, x0, sigma, noise, cond_shuf, patch).cpu()
        if (start // batch_size) % 20 == 0:
            log(f"  {arm_name(cfg)}: {start + len(idx)}/{n}")
    return arm_name(cfg), val["keys"][:n], losses, shuffled


def paired(a, b, n_boot=10_000, seed=0):
    d = a - b
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(d), (n_boot, len(d)), generator=g)
    boots = d[idx].mean(1)
    return {"mean_diff": d.mean().item(), "ci95": [boots.quantile(0.025).item(), boots.quantile(0.975).item()],
            "frac_a_better": (d < 0).float().mean().item(), "rel_diff": (d.mean() / b.mean()).item()}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    from diffusers import WanTransformer3DModel

    base = WanTransformer3DModel.from_pretrained(args.wan_dir, subfolder="transformer", torch_dtype=torch.bfloat16)
    base.requires_grad_(False)
    per_arm, keys = {}, None
    for d in args.arm_dirs:
        name, k, losses, shuffled = score_arm(d, args.wan_dir, args.batch_size, args.n_val, base, device)
        if keys is not None and k != keys:
            sys.exit(f"val clip order differs for {d}; cannot pair")
        keys, per_arm[name] = k, losses
        if shuffled is not None:
            per_arm[f"{name}:shuffled"] = shuffled
            r = paired(losses.mean(1), shuffled.mean(1))
            log(f"{name}: real vs shuffled conditioning diff={r['mean_diff']:+.5f} "
                f"CI95=[{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}] real better on {100 * r['frac_a_better']:.1f}% of clips")
        log(f"{name}: mean val loss {losses.mean():.5f}  per sigma "
            + " ".join(f"{s}:{v:.4f}" for s, v in zip(EVAL_SIGMAS, losses.mean(0).tolist())))
    torch.save({"keys": keys, "sigmas": EVAL_SIGMAS, "losses": per_arm}, os.path.join(args.output_dir, "per_clip_losses.pt"))

    summary = {"means": {n: l.mean().item() for n, l in per_arm.items()}, "pairs": {}}
    rows = []
    for a, b in itertools.permutations(per_arm, 2):
        own_shuffle = b == f"{a}:shuffled"
        both_real = ":shuffled" not in a and ":shuffled" not in b
        if own_shuffle or (both_real and (a < b or "null" in (a, b))) or (b.startswith("null") and ":shuffled" in a):
            r = paired(per_arm[a].mean(1), per_arm[b].mean(1))
            summary["pairs"][f"{a} vs {b}"] = r
            rows.append([a, b, r["mean_diff"], r["ci95"][0], r["ci95"][1], r["frac_a_better"], r["rel_diff"]])
            sig = "REAL" if r["ci95"][1] < 0 or r["ci95"][0] > 0 else "n.s."
            log(f"{a:>28} vs {b:<28} diff={r['mean_diff']:+.5f} CI95=[{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}] "
                f"a better on {100 * r['frac_a_better']:.1f}% of clips  ({100 * r['rel_diff']:+.2f}%)  {sig}")
    json.dump(summary, open(os.path.join(args.output_dir, "comparison.json"), "w"), indent=2)
    flat = {f"mean/{n}": v for n, v in summary["means"].items()}
    log_summary(args.wandb_project, "vdm-pilot-paired-comparison", {"arm_dirs": args.arm_dirs}, flat,
                tables={"paired": (["arm_a", "arm_b", "mean_diff", "ci_lo", "ci_hi", "frac_a_better", "rel_diff"], rows)},
                job_type="compare")


if __name__ == "__main__":
    main()
