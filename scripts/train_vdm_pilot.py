"""VDM conditioning pilot: does the VLM latent tell Wan more about the future than its caption?

Trains Wan2.1-T2V-1.3B (LoRA on attention) to generate a Future-L1-50K example's future
from its context frame, with one of four conditioning arms -- everything else identical:

  null      empty-prompt umT5 embedding (context frame only)
  caption   umT5 embedding of the Stage 0 VLM's own generated answer
  latent    the same VLM pass's 4 latent-slot hidden states -> trainable projector -> 4 tokens
  both      caption tokens followed by the latent tokens
  qwen      the same VLM's hidden states over its own answer tokens (vdm_pilot_qwen_states.py,
            --qwen_layer last|mid) -> trainable per-token projector: the caption's information
            without the decode-to-text / umT5 re-encode bottleneck

Inputs are precomputed (vdm_pilot_vlm_outputs.py, vdm_pilot_encode.py): normalized Wan
VAE latents of a 17-frame clip (frame 0 = context frame, 1..16 = time-lapsed future),
caption embeddings, and VLM latents, joined on "<video>|<row>".

Objective: Wan's rectified-flow loss. x_t = (1 - s) x0 + s e, target v = e - x0,
s from shifted-uniform sampling (shift 3, Wan's 480p setting). Latent frame 0 (the
context frame) is kept clean with per-token timestep 0 -- Wan's transformer takes a
(batch, seq_len) timestep -- and excluded from the loss, the same prefix-conditioning
VANS uses. Conditioning sequences are zero-padded to 512 tokens like WanPipeline.

Bridge distillation (--distill_steps N, for cond qwen/latent; added 2026-10-06): the first
pilot showed every arm with a freshly initialized projector starts far worse than null and
spends the whole run recovering to null -- the 0.5%-of-loss conditioning signal is too weak
to train a bridge from scratch on 6k clips. With distillation, the projector is first
trained alone (LoRA off, Wan frozen) so that Wan's velocity prediction from the projected
states matches its prediction from the umT5 caption embedding of the same example, on the
same noisy input -- a dense signal, and a fair start since both carry the same answer.
The usual LoRA + flow-loss training then runs for --max_steps as in every other arm.

Metric: held-out flow loss on a fixed set of validation clips at a fixed grid of noise
levels with fixed noise (seeded per clip), so numbers are directly comparable across
arms and steps. Lower = the conditioning carries more information about the real future.
Logged per noise level to metrics.jsonl and W&B.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_TEXT_LEN = 512
WAN_TEXT_DIM = 4096
EVAL_SIGMAS = (0.1, 0.3, 0.5, 0.7, 0.9)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cond", choices=["null", "caption", "latent", "both", "qwen"], required=True)
    p.add_argument("--qwen_layer", choices=["last", "mid"], default="last", help="--cond qwen: which saved layer")
    p.add_argument("--wan_dir", required=True)
    p.add_argument("--encoded_dir", required=True, help="vdm_pilot_encode.py output dir")
    p.add_argument("--vlm_dir", required=True, help="vdm_pilot_vlm_outputs.py output dir (latents)")
    p.add_argument("--text_tag", default="caption")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--cond_dropout", type=float, default=0.1, help="replace conditioning with null (for CFG)")
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_n", type=int, default=256)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--distill_steps", type=int, default=0,
                   help="cond qwen/latent: projector-only distillation steps toward the caption-conditioned "
                        "prediction before normal training (0 = off, the original pilot)")
    p.add_argument("--distill_lr", type=float, default=3e-4)
    p.add_argument("--distill_eval_every", type=int, default=500)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


class LatentProjector(nn.Module):
    """VLM latent slots (n, 2048) -> Wan conditioning tokens (n, 4096). The final LayerNorm's gain
    starts at the caption embeddings' per-dim RMS so the projected tokens enter Wan's text
    embedder at the scale it was trained on."""

    def __init__(self, in_dim, out_dim, init_rms):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(),
                                 nn.Linear(out_dim, out_dim))
        self.norm = nn.LayerNorm(out_dim)
        with torch.no_grad():
            self.norm.weight.fill_(init_rms)

    def forward(self, x):
        return self.norm(self.net(x))


def load_split(args, split):
    vid = torch.load(os.path.join(args.encoded_dir, f"{split}_video.pt"))
    vlm = torch.load(os.path.join(args.vlm_dir, f"{split}_vlm.pt"))
    txt_path = os.path.join(args.encoded_dir, f"{split}_text_{args.text_tag}.pt")
    txt = torch.load(txt_path) if os.path.exists(txt_path) else None
    vi = {k: j for j, k in enumerate(vid["keys"])}
    li = {k: j for j, k in enumerate(vlm["keys"])}
    ti = {k: j for j, k in enumerate(txt["keys"])} if txt else None
    qwen = None
    if getattr(args, "cond", None) == "qwen":
        qwen = torch.load(os.path.join(args.vlm_dir, f"{split}_qwen_states.pt"))
        qi = {k: j for j, k in enumerate(qwen["keys"])}
    keys = [k for k in vid["keys"] if k in li and (ti is None or k in ti) and (qwen is None or k in qi)]
    data = {
        "keys": keys,
        "video": vid["latents"][[vi[k] for k in keys]],
        "latent": vlm["latents"][[li[k] for k in keys]],
        "text": [txt["embeds"][ti[k]] for k in keys] if txt else None,
        "null": txt["null"] if txt else None,
        "qwen": [qwen[getattr(args, "qwen_layer", "last")][qi[k]] for k in keys] if qwen else None,
    }
    log(f"{split}: {len(keys)} examples (video {len(vid['keys'])}, vlm {len(vlm['keys'])}, "
        f"text {len(txt['keys']) if txt else 'n/a'})")
    return data


def build_cond(args, data, idx, projector, null_emb, device, drop_mask=None):
    """List of (L_i, 4096) conditioning sequences for batch indices idx, zero-padded to MAX_TEXT_LEN."""
    seqs = []
    lat_tok = None
    if args.cond in ("latent", "both"):
        lat_tok = projector(data["latent"][idx].to(device).float())  # (B, 4, 4096)
    for b, j in enumerate(idx):
        if args.cond == "null" or (drop_mask is not None and drop_mask[b]):
            s = null_emb
        elif args.cond == "caption":
            s = data["text"][j].to(device).float()
        elif args.cond == "latent":
            s = lat_tok[b]
        elif args.cond == "qwen":
            s = projector(data["qwen"][j].to(device).float())
        else:
            s = torch.cat([data["text"][j].to(device).float(), lat_tok[b]], dim=0)
        s = s[:MAX_TEXT_LEN]
        seqs.append(F.pad(s, (0, 0, 0, MAX_TEXT_LEN - s.shape[0])))
    return torch.stack(seqs)


def flow_inputs(x0, sigma, noise, patch):
    """Noisy latents with frame 0 clean, and the per-token timestep (frame-0 tokens at 0)."""
    s = sigma.view(-1, 1, 1, 1, 1)
    xt = (1 - s) * x0 + s * noise
    xt[:, :, 0] = x0[:, :, 0]
    b, _, f, h, w = x0.shape
    tokens_per_frame = (h // patch[1]) * (w // patch[2])
    t = (sigma * 1000).view(b, 1).expand(b, (f // patch[0]) * tokens_per_frame).clone()
    t[:, :tokens_per_frame] = 0
    return xt, t


def predict(model, xt, t, cond):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(hidden_states=xt.to(torch.bfloat16), timestep=t, encoder_hidden_states=cond.to(torch.bfloat16),
                     return_dict=False)[0]


def flow_loss(model, x0, sigma, noise, cond, patch):
    xt, t = flow_inputs(x0, sigma, noise, patch)
    pred = predict(model, xt, t, cond)
    target = noise - x0
    return F.mse_loss(pred.float()[:, :, 1:], target[:, :, 1:], reduction="none").mean(dim=(1, 2, 3, 4))


@torch.no_grad()
def evaluate(args, model, projector, val, null_emb, device, patch):
    model.eval()
    if projector is not None:
        projector.eval()
    n = min(args.eval_n, len(val["keys"]))
    per_sigma = {s: [] for s in EVAL_SIGMAS}
    for start in range(0, n, args.batch_size):
        idx = list(range(start, min(start + args.batch_size, n)))
        x0 = val["video"][idx].to(device).float()
        cond = build_cond(args, val, idx, projector, null_emb, device)
        for s in EVAL_SIGMAS:
            gens = [torch.Generator(device="cpu").manual_seed(10_000 * j + int(s * 100)) for j in idx]
            noise = torch.stack([torch.randn(x0.shape[1:], generator=g) for g in gens]).to(device)
            sigma = torch.full((len(idx),), s, device=device)
            per_sigma[s] += flow_loss(model, x0, sigma, noise, cond, patch).tolist()
    model.train()
    if projector is not None:
        projector.train()
    out = {f"val_loss_s{s}": sum(v) / len(v) for s, v in per_sigma.items()}
    out["val_loss"] = sum(out.values()) / len(out)
    return out


def distill_bridge(args, model, projector, train, val, null_emb, device, patch, record):
    """Projector-only phase: match Wan's caption-conditioned velocity prediction (see module docstring)."""
    teacher_args = argparse.Namespace(**{**vars(args), "cond": "caption"})
    opt = torch.optim.AdamW(projector.parameters(), lr=args.distill_lr, weight_decay=args.weight_decay)
    model.disable_adapters()
    n_train = len(train["keys"])
    t0 = time.time()
    for step in range(1, args.distill_steps + 1):
        idx = random.sample(range(n_train), args.batch_size)
        x0 = train["video"][idx].to(device).float()
        u = torch.rand(len(idx), device=device)
        sigma = args.flow_shift * u / (1 + (args.flow_shift - 1) * u)
        xt, t = flow_inputs(x0, sigma, torch.randn_like(x0), patch)
        with torch.no_grad():
            teacher = predict(model, xt, t, build_cond(teacher_args, train, idx, None, null_emb, device)).float()
        student = predict(model, xt, t, build_cond(args, train, idx, projector, null_emb, device)).float()
        loss = F.mse_loss(student[:, :, 1:], teacher[:, :, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            dt = (time.time() - t0) / 50
            t0 = time.time()
            log(f"distill {step}/{args.distill_steps} match_loss={loss.item():.6f} grad_norm={gn.item():.3f} ({dt:.2f}s/step)")
            record({"distill/match_loss": loss.item(), "distill/grad_norm": gn.item()}, step, distill=True)
        if step % args.distill_eval_every == 0 or step == args.distill_steps:
            ev = evaluate(args, model, projector, val, null_emb, device, patch)
            log(f"[distill eval] step {step}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
            record({f"distill/{k}": v for k, v in ev.items()}, step, distill=True)
    model.enable_adapters()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    from diffusers import WanTransformer3DModel
    from peft import LoraConfig

    train, val = load_split(args, "train"), load_split(args, "val")
    if args.cond in ("caption", "both") and train["text"] is None:
        sys.exit("caption arms need <encoded_dir>/{split}_text_<tag>.pt (vdm_pilot_encode.py --what text)")
    null_path = os.path.join(args.encoded_dir, f"val_text_{args.text_tag}.pt")
    null_emb = (train["null"] if train["null"] is not None else torch.load(null_path)["null"]).to(device).float()

    model = WanTransformer3DModel.from_pretrained(args.wan_dir, subfolder="transformer", torch_dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, init_lora_weights=True,
                                 target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    for p_ in model.parameters():
        if p_.requires_grad:
            p_.data = p_.data.float()
    model.enable_gradient_checkpointing()
    model.to(device).train()
    patch = tuple(model.config.patch_size)

    projector = None
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    if args.cond in ("latent", "both", "qwen"):
        texts = train["text"] or [null_emb.cpu()]
        rms = torch.cat([t.float() for t in texts[:500]]).pow(2).mean().sqrt().item()
        projector = LatentProjector(train["latent"].shape[-1], WAN_TEXT_DIM, rms).to(device)
        params += list(projector.parameters())
        log(f"latent projector: init output RMS {rms:.4f}")
    log(f"trainable params: {sum(p_.numel() for p_ in params) / 1e6:.2f}M  cond={args.cond}  patch={patch}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) * 0.5 * (
        1 + math.cos(math.pi * min(s, args.max_steps) / args.max_steps)))

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args),
                                   name=args.wandb_run_name or f"vdm-pilot-{args.cond}-{os.environ.get('SLURM_JOB_ID', 'local')}")
            log(f"W&B: {wandb_run.url}")
        except Exception as e:
            log(f"WARNING: W&B init failed ({e}); continuing without it")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    def record(row, step, distill=False):
        """metrics.jsonl keeps the training step (distill rows are tagged); W&B needs one increasing
        step axis, so training steps are shifted past the distillation phase there."""
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "phase": "distill" if distill else "train", **row,
                                "time": time.time()}) + "\n")
        nonlocal wandb_run
        if wandb_run is not None:
            try:
                wandb_run.log(row, step=step if distill else step + args.distill_steps + 1)
            except Exception as e:
                log(f"WARNING: wandb.log failed ({e}); disabling W&B")
                wandb_run = None

    def save(step):
        d = os.path.join(args.output_dir, f"step_{step}")
        os.makedirs(d, exist_ok=True)
        from peft.utils import get_peft_model_state_dict
        torch.save(get_peft_model_state_dict(model), os.path.join(d, "wan_lora.pt"))
        if projector is not None:
            torch.save(projector.state_dict(), os.path.join(d, "latent_projector.pt"))
        json.dump(vars(args), open(os.path.join(d, "config.json"), "w"), indent=2)
        log(f"saved {d}")

    if args.distill_steps:
        if projector is None or train["text"] is None:
            sys.exit("--distill_steps needs a projector arm (qwen/latent) and caption embeddings")
        distill_bridge(args, model, projector, train, val, null_emb, device, patch, record)
    ev = evaluate(args, model, projector, val, null_emb, device, patch)
    log(f"[eval] step 0: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
    record(ev, 0)

    n_train = len(train["keys"])
    order, pos = list(range(n_train)), n_train
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        if pos + args.batch_size > n_train:
            random.shuffle(order)
            pos = 0
        idx = order[pos:pos + args.batch_size]
        pos += args.batch_size
        x0 = train["video"][idx].to(device).float()
        drop = torch.rand(len(idx)) < args.cond_dropout
        cond = build_cond(args, train, idx, projector, null_emb, device, drop_mask=drop)
        u = torch.rand(len(idx), device=device)
        sigma = args.flow_shift * u / (1 + (args.flow_shift - 1) * u)
        noise = torch.randn_like(x0)
        loss = flow_loss(model, x0, sigma, noise, cond, patch).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0:
            dt = (time.time() - t0) / args.log_every
            t0 = time.time()
            log(f"step {step}/{args.max_steps} loss={loss.item():.4f} grad_norm={gn.item():.3f} "
                f"lr={sched.get_last_lr()[0]:.2e} ({dt:.2f}s/step)")
            record({"train_loss": loss.item(), "grad_norm": gn.item(), "lr": sched.get_last_lr()[0],
                    "seconds_per_step": dt}, step)
        if step % args.eval_every == 0 or step == args.max_steps:
            ev = evaluate(args, model, projector, val, null_emb, device, patch)
            log(f"[eval] step {step}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
            record(ev, step)
        if step % args.save_every == 0 or step == args.max_steps:
            save(step)
    if wandb_run is not None:
        wandb_run.finish()
    log("VDM PILOT TRAINING COMPLETE")


if __name__ == "__main__":
    main()
