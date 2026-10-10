"""Core-claim test in the VANS setting: does a latent from the VLM, trained end to end with the video
model's loss, add to the caption?

Two arms, trained identically (same data order, steps, lr, seed), differing only in conditioning:
  caption  DiT cross-attends to umT5(caption)                              -- what VANS does
  latent   DiT cross-attends to umT5(caption) followed by K latent tokens

The latent: a Qwen2.5-VL-3B copy (initialized from VANS's VLM LoRA, LoRA trainable) reads
  [system] [input video] [instruction] -> [Think]think[/Think][Ans]caption[/Ans] <BOQ><Q0>..<Q(K-1)><EOQ>
The query tokens use unused rows of Qwen's embedding table (ids >= len(tokenizer)); their input
embeddings are replaced by a learnable (K, 2048) parameter through a hook. The final hidden states at
the K query positions go through a projector (LayerNorm -> MLP -> RMS-matched to umT5 embeddings) to
K x 4096 tokens, appended right after the caption's tokens in the 512-token text context. The VDM flow
loss trains the DiT, the projector, the query embeddings and the Qwen LoRA together.

Setting (both arms): VANS's (6 reference latents of the input video, 33 frames at 352x640, the
next-event clip as target, all DiT blocks trained), DiT initialized from the released VANS DiT, our
data/vans_train rows, ground-truth think + caption during training (teacher forcing, as VANS SFT).
At test time the captions for both arms come from the frozen VANS VLM (or the GT, for the oracle
condition); the latent arm's Qwen copy then reads that answer and produces the latent.

  train:    --mode train --arm caption|latent
  cache:    --mode cache --shard k --num_shards n   (pre-encode training rows into --cache_dir; no training)
  generate: --mode generate --arm ... --ckpt <step dir> --captions vans|oracle   (VANS-benchmark answers)
"""
import argparse
import glob
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from vans_rl_common import (NEGATIVE, build_vdm, build_vlm, decode, encode_text, encode_video, load_frames, log,
                            lora_state_dict, ref_latents, sample_ode, velocity)

MAX_TEXT = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "generate", "cache"], required=True)
    p.add_argument("--arm", choices=["caption", "latent"], default="caption")
    p.add_argument("--data_dir", help="train: data/vans_train")
    p.add_argument("--bench_dir", help="generate: data/vans_eval")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vans_code", required=True)
    p.add_argument("--vans_dir", required=True, help="released VANS weights (DiT init, VLM LoRA init)")
    p.add_argument("--wan_orig_dir", required=True)
    p.add_argument("--wan_diffusers_dir", required=True)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--num_queries", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-5, help="DiT and Qwen LoRA")
    p.add_argument("--lr_new", type=float, default=1e-4, help="projector and query embeddings")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--cache_dir", default=None, help="shared cache of ref/target latents + caption embeddings")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    # generate
    p.add_argument("--ckpt", default=None)
    p.add_argument("--captions", choices=["vans", "oracle"], default="vans")
    p.add_argument("--vans_captions", default=None, help="glob of results/vans_eval/vans/captions_shard*.jsonl")
    p.add_argument("--name", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


class Projector(nn.Module):
    def __init__(self, d_in, d_out, target_rms):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        self.gain = nn.Parameter(torch.tensor(float(target_rms)))

    def forward(self, h):
        y = self.mlp(self.norm(h.float()))
        return y / y.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * self.gain


class LatentQwen(nn.Module):
    """Qwen copy + K query tokens + projector."""

    def __init__(self, args, device, target_rms):
        super().__init__()
        self.agent = build_vlm(args, device, os.path.join(args.vans_dir, "VANS_mllm.safetensors"))
        self.agent.train()
        for m in self.agent.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        bb = self.agent.mllm_backbone
        bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        bb.enable_input_require_grads()
        for n_, p_ in bb.named_parameters():
            if "lora_" in n_:
                p_.data = p_.data.float()
        tok = self.agent.tokenizer.tokenizer
        base_vocab = len(tok)
        hidden = bb.get_base_model().config.hidden_size
        n_rows = bb.get_base_model().get_input_embeddings().num_embeddings
        k = args.num_queries
        if base_vocab + k + 2 > n_rows:
            raise RuntimeError("not enough unused embedding rows for the query tokens")
        self.boq, self.eoq = base_vocab, base_vocab + 1
        self.qids = torch.arange(base_vocab + 2, base_vocab + 2 + k)
        self.query = nn.Parameter(torch.randn(k, hidden) * 0.02)
        self.proj = Projector(hidden, 4096, target_rms)
        self.query.data = self.query.data.to(device)
        self.proj.to(device)
        emb = bb.get_base_model().get_input_embeddings()
        qids = self.qids.to(device)

        def hook(module, inputs, out):
            ids = inputs[0]
            hit = (ids[..., None] == qids).any(-1)
            if not hit.any():
                return out
            idx = (ids[hit][:, None] == qids).float().argmax(-1)
            out = out.clone()
            out[hit] = self.query[idx].to(out.dtype)
            return out

        emb.register_forward_hook(hook)
        self.device = device

    def trainable(self):
        lora = [p for n, p in self.agent.mllm_backbone.named_parameters() if "lora_" in n]
        for p in lora:
            p.requires_grad_(True)
        return lora, [self.query] + list(self.proj.parameters())

    def forward(self, video_path, instruction, answer_text):
        """answer_text: the assistant answer ('[Think]..[/Think][Ans]..[/Ans]'). Returns (1, K, 4096)."""
        a = self.agent
        prompt, _, _ = a.tokenize(a.tokenizer, is_train=False, add_query=False, instructions=instruction,
                                  caption="", video=video_path, device=self.device, torch_dtype=torch.bfloat16)
        prompt = {k: v for k, v in prompt.items() if torch.is_tensor(v) and k != "labels"}
        tok = a.tokenizer.tokenizer
        ans = tok(answer_text, add_special_tokens=False, return_tensors="pt").input_ids[0][:768].to(self.device)
        q = torch.cat([torch.tensor([self.boq]), self.qids, torch.tensor([self.eoq])]).to(self.device)
        ids = torch.cat([prompt["input_ids"], ans[None], q[None]], dim=1)
        mask = torch.ones_like(ids)
        kw = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}
        base = a.mllm_backbone.get_base_model()
        head = base.lm_head
        base.lm_head = nn.Identity()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = a.mllm_backbone(input_ids=ids, attention_mask=mask, **kw).logits
        finally:
            base.lm_head = head
        h = hidden[:, -(len(self.qids) + 1):-1]  # the K query positions
        return self.proj(h)


def with_latent(ctx, lat):
    """Insert K latent tokens right after the caption's (non-zero) tokens of a (1, 512, 4096) context."""
    n = int((ctx[0].abs().sum(-1) > 0).sum())
    k = lat.shape[1]
    n = min(n, MAX_TEXT - k)
    out = torch.cat([ctx[:, :n].to(lat.dtype), lat, ctx[:, n + k:].to(lat.dtype)], dim=1)
    return out[:, :MAX_TEXT]


def answer_text(think, caption):
    return f"[Think]{think}[/Think][Ans]{caption}[/Ans]"


def make_encoder(args, pipe, device):
    """encode_row(r) -> {ref, x0, ctx}, reading/writing args.cache_dir (shared by both arms)."""
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    def encode_row(r):
        path = os.path.join(args.cache_dir, f"{r['sample_id']}.pt") if args.cache_dir else None
        if path and os.path.exists(path):
            c = torch.load(path, map_location=device)
            c["ctx"] = F.pad(c["ctx"], (0, 0, 0, c["ctx_len"] - c["ctx"].shape[1]))
            return c
        c = {"ref": ref_latents(encode_video(pipe, load_frames(os.path.join(args.data_dir, r["input_video"])))),
             "x0": encode_video(pipe, load_frames(os.path.join(args.data_dir, r["target_video"]))),
             "ctx": encode_text(pipe, r["gt_caption"])}
        if path:
            n = int((c["ctx"][0].abs().sum(-1) > 0).sum())
            torch.save({"ref": c["ref"].cpu(), "x0": c["x0"].cpu(), "ctx": c["ctx"][:, :n].cpu(),
                        "ctx_len": c["ctx"].shape[1]}, f"{path}.{os.getpid()}.tmp")
            os.replace(f"{path}.{os.getpid()}.tmp", path)  # atomic: concurrent jobs may encode the same row
        return c

    return encode_row


def cache(args):
    device = "cuda"
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "benchmark.jsonl"))][args.shard::args.num_shards]
    pipe = build_vdm(args, device, os.path.join(args.vans_dir, "VANS_vlm.safetensors"))
    encode_row = make_encoder(args, pipe, device)
    t0, n_fail = time.time(), 0
    for i, r in enumerate(rows):
        try:
            encode_row(r)
        except Exception as e:
            n_fail += 1
            log(f"WARNING: {r['sample_id']}: {str(e)[-300:]}")
        if (i + 1) % 50 == 0:
            log(f"cache shard {args.shard}: {i + 1}/{len(rows)} ({(time.time() - t0) / (i + 1):.1f}s/row, {n_fail} failed)")
    log(f"CACHE DONE: {len(rows)} rows, {n_fail} failed")


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "benchmark.jsonl"))]
    log(f"{len(rows)} training rows, arm {args.arm}")

    pipe = build_vdm(args, device, os.path.join(args.vans_dir, "VANS_vlm.safetensors"))
    dit = pipe.dit
    dit.float().train().requires_grad_(True)
    groups = [{"params": list(dit.parameters()), "lr": args.lr}]
    lq = None
    if args.arm == "latent":
        rms = encode_text(pipe, rows[0]["gt_caption"]).float()
        n = int((rms[0].abs().sum(-1) > 0).sum())
        target_rms = rms[0, :n].pow(2).mean().sqrt().item()
        lq = LatentQwen(args, device, target_rms)
        lora, new = lq.trainable()
        groups += [{"params": lora, "lr": args.lr}, {"params": new, "lr": args.lr_new}]
        log(f"latent: {args.num_queries} queries, umT5 token RMS {target_rms:.4f}")
    params = [p for g in groups for p in g["params"]]
    log(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
    sch = pipe.scheduler
    sch.set_timesteps(1000, training=True, shift=5.0)
    encode_row = make_encoder(args, pipe, device)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args),
                                   name=args.wandb_run_name or f"vans-latent-{args.arm}-{os.environ.get('SLURM_JOB_ID', 'local')}")
        except Exception as e:
            log(f"WARNING: W&B init failed ({e})")

    def save(step):
        from safetensors.torch import save_file
        d = os.path.join(args.output_dir, f"step_{step}")
        os.makedirs(d, exist_ok=True)
        save_file({k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in dit.state_dict().items()},
                  os.path.join(d, "dit.safetensors"))
        if lq is not None:
            save_file({k: v.contiguous() for k, v in lora_state_dict(lq.agent).items()},
                      os.path.join(d, "latent_vlm_lora.safetensors"))
            torch.save({"query": lq.query.detach().cpu(), "proj": lq.proj.state_dict(),
                        "num_queries": args.num_queries}, os.path.join(d, "latent_head.pt"))
        log(f"saved {d}")

    rng = random.Random(args.seed)  # same data order and timesteps for both arms
    order, pos, t0, run, fails = list(range(len(rows))), len(rows), time.time(), 0.0, 0
    for step in range(1, args.max_steps + 1):
        if pos >= len(rows):
            rng.shuffle(order)
            pos = 0
        r = rows[order[pos]]
        pos += 1
        tid = rng.randrange(len(sch.timesteps))
        g = torch.Generator("cpu").manual_seed(args.seed * 100003 + step)
        try:
            c = encode_row(r)
            x0 = c["x0"].float()
            ctx = c["ctx"]
            if lq is not None:
                lat = lq(os.path.join(args.data_dir, r["input_video"]), r["instruction"],
                         answer_text(r["think"], r["gt_caption"]))
                ctx = with_latent(ctx, lat)
            t = sch.timesteps[tid].to(device)
            noise = torch.randn(x0.shape, generator=g).to(device)
            xt = sch.add_noise(x0, noise, t)
            pred = velocity(pipe, dit, xt.to(torch.bfloat16), c["ref"], t, ctx.to(torch.bfloat16), grad_ckpt=True)
            loss = F.mse_loss(pred.float(), sch.training_target(x0, noise, t).float()) * sch.training_weight(t).to(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            fails = 0
        except Exception as e:
            fails += 1
            log(f"WARNING: {r['sample_id']}: {str(e)[-300:]}")
            if fails >= 20:
                raise
            continue
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        if step % args.log_every == 0:
            dt = (time.time() - t0) / args.log_every
            row = {"loss": run / args.log_every, "grad_norm": gn.item(), "seconds_per_step": dt}
            if lq is not None:
                row["latent_gain"] = lq.proj.gain.item()
            log(f"step {step}/{args.max_steps} " + " ".join(f"{k}={v:.4f}" for k, v in row.items()))
            with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, **row}) + "\n")
            if wandb_run:
                wandb_run.log(row, step=step)
            t0, run = time.time(), 0.0
        if step % args.save_every == 0 or step == args.max_steps:
            save(step)
    log("TRAINING COMPLETE")


@torch.no_grad()
def generate(args):
    from vans_rl_common import vans_imports
    vans_imports(args.vans_code)
    from vans import save_video
    device = "cuda"
    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    rows = rows[:args.limit] if args.limit else rows
    rows = rows[args.shard::args.num_shards]
    answers = {}
    if args.captions == "vans":
        for f in sorted(glob.glob(args.vans_captions)):
            for l in open(f):
                c = json.loads(l)
                answers[c["sample_id"]] = (c["raw"] or answer_text("", c["caption"]), c["caption"])
    else:
        answers = {r["sample_id"]: (answer_text(r["think"], r["gt_caption"]), r["gt_caption"]) for r in rows}
    rows = [r for r in rows if r["sample_id"] in answers]
    name = args.name or f"latent_{args.arm}_{args.captions}"
    out = os.path.join(args.output_dir, name)
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    cap_path = os.path.join(out, f"captions_shard{args.shard}.jsonl")
    done = {json.loads(l)["sample_id"] for l in open(cap_path)} if os.path.exists(cap_path) else set()

    pipe = build_vdm(args, device, os.path.join(args.ckpt, "dit.safetensors"))
    pipe.dit.eval()
    lq = None
    if args.arm == "latent":
        head = torch.load(os.path.join(args.ckpt, "latent_head.pt"), map_location="cpu")
        args.num_queries = head["num_queries"]
        lq = LatentQwen(args, device, 1.0)
        from safetensors.torch import load_file
        lq.agent.load_state_dict({k: v.to(device) for k, v in
                                  load_file(os.path.join(args.ckpt, "latent_vlm_lora.safetensors")).items()}, strict=False)
        lq.query.data = head["query"].to(device)
        lq.proj.load_state_dict(head["proj"])
        lq.eval()
    neg = encode_text(pipe, NEGATIVE, positive=False)
    for i, r in enumerate(rows):
        if r["sample_id"] in done:
            continue
        t0 = time.time()
        raw, caption = answers[r["sample_id"]]
        in_path = os.path.join(args.bench_dir, r["input_video"])
        ref = ref_latents(encode_video(pipe, load_frames(in_path)))
        ctx = encode_text(pipe, caption)
        if lq is not None:
            ctx = with_latent(ctx, lq(in_path, r["instruction"], raw)).to(torch.bfloat16)
        lat = sample_ode(pipe, pipe.dit, ref, ctx, neg, steps=50, cfg=5.0, shift=5.0, seed=0)
        save_video(decode(pipe, lat), os.path.join(out, "videos", f"{r['sample_id']}.mp4"), fps=11, quality=5)
        with open(cap_path, "a") as f:
            f.write(json.dumps({"sample_id": r["sample_id"], "caption": caption}) + "\n")
        log(f"[{name} shard {args.shard}] {i + 1}/{len(rows)} {r['sample_id']} ({time.time() - t0:.0f}s)")
    log("GENERATION DONE")


if __name__ == "__main__":
    a = parse_args()
    {"train": train, "generate": generate, "cache": cache}[a.mode](a)
