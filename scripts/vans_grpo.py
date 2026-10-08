"""Joint-GRPO (VANS, arXiv 2511.16669 section 4), reimplemented on top of code/VANS.

Stage 1 -- visualization-friendly VLM tuning (--stage 1). VDM frozen. For each prompt (input video +
  instruction) the VLM samples G answers; each answer's caption is rendered by the frozen VDM, and
      r1 = r_format + ROUGE-L(caption, GT caption) + CLIP-V(rendered video, GT video)      (all weights 1)
  GRPO on the VLM LoRA: group-normalized advantages, PPO-clipped ratio, k3 KL to the reference
  policy (the SFT LoRA, kept frozen as a second adapter-free copy is unnecessary: the reference is
  the policy with the stage-start LoRA weights, stored once), beta 0.004. Paper: 800 steps, lr 5e-5.

Stage 2 -- context-faithful VDM adaptation (--stage 2). VLM frozen (stage-1 result). For each prompt
  the VLM samples an anchor caption, kept only if ROUGE-L(anchor, GT) >= 0.6. The VDM samples G
  videos for the anchor with Flow-GRPO's SDE sampler, and
      r2 = CLIP-V(video, GT video) + CLIPScore(video, anchor)                                 (weights 1)
  Flow-GRPO update on the DiT (all parameters by default, as in VANS SFT; --vdm_lora_rank for LoRA):
  per-step Gaussian log-prob ratios, clip range 1e-3, KL to the stage-start DiT, beta 0.004.
  Paper: 1K steps, lr 5e-5.

Group size 8 (paper). Not stated in the paper and set here: VLM sampling temperature 1.0, reward
renders with --reward_steps ODE steps, SDE rollouts with --sde_steps steps and eta 0.7 (Flow-GRPO's
default), one optimizer step per --timesteps_per_update SDE steps. Multi-GPU: one process per GPU
(torchrun), each takes different prompts, gradients averaged.
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from vans_rl_common import (NEGATIVE, ClipScorer, build_vdm, build_vlm, decode, encode_text, encode_video,
                            guided_velocity, load_frames, log, lora_state_dict, parse_answer, ref_latents,
                            rouge_l, sample_ode, sample_sde, sde_step)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--data_dir", required=True, help="RL prompts (benchmark.jsonl format)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vans_code", required=True)
    p.add_argument("--wan_orig_dir", required=True)
    p.add_argument("--wan_diffusers_dir", required=True)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--vlm_lora", required=True, help="stage 1: SFT VLM LoRA to start from; stage 2: stage-1 result")
    p.add_argument("--dit", required=True, help="DiT weights (SFT dit.safetensors)")
    p.add_argument("--num_prompts", type=int, default=1000, help="paper: 1K manually selected RL samples")
    p.add_argument("--max_steps", type=int, default=800)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.004)
    p.add_argument("--clip_range", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--reward_steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sde_steps", type=int, default=16)
    p.add_argument("--eta", type=float, default=0.7)
    p.add_argument("--timesteps_per_update", type=int, default=5)
    p.add_argument("--anchor_rouge", type=float, default=0.6)
    p.add_argument("--anchor_tries", type=int, default=4)
    p.add_argument("--vdm_lora_rank", type=int, default=0)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return rank, dist.get_world_size()
    return 0, 1


def allreduce_grads(params, world):
    if world > 1:
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world


def allmean(x, world):
    if world == 1:
        return x
    t = torch.tensor(x, device="cuda", dtype=torch.float64)
    dist.all_reduce(t)
    return (t / world).item()


def advantages(r):
    r = np.asarray(r, dtype=np.float64)
    return (r - r.mean()) / (r.std() + 1e-4)


# ---------------------------------------------------------------- VLM side
def vlm_prompt(agent, row, data_dir, device):
    inputs, _, _ = agent.tokenize(agent.tokenizer, is_train=False, add_query=False, instructions=row["instruction"],
                                  caption="", video=os.path.join(data_dir, row["input_video"]), device=device,
                                  torch_dtype=torch.bfloat16)
    # model inputs only: VANS's tokenize also returns its own labels and non-tensor metadata
    return {k: v for k, v in inputs.items() if torch.is_tensor(v) and k != "labels"}


@torch.no_grad()
def vlm_sample(agent, inputs, n, temperature, max_new_tokens):
    # Qwen2.5-VL packs video patches as (n_patches, dim) with one grid row per video, so repeating
    # every tensor along dim 0 gives n copies of the same prompt
    rep = {k: v.repeat(n, *[1] * (v.ndim - 1)) for k, v in inputs.items()}
    was_training = agent.training
    agent.eval()  # checkpointing (train mode) would force use_cache off; dropout is 0 either way
    try:
        out = agent.mllm_backbone.generate(**rep, max_new_tokens=max_new_tokens, do_sample=True,
                                           temperature=temperature, top_p=1.0, use_cache=True,
                                           return_dict_in_generate=True)
    finally:
        agent.train(was_training)
    plen = inputs["input_ids"].shape[1]
    seqs = out.sequences[:, plen:]
    texts = agent.tokenizer.batch_decode(seqs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return seqs, texts


def response_logprobs(model, inputs, resp_ids, pad_id):
    """Per-token log-probs of one response (T,) given the prompt inputs (batch 1). The LM head runs only on
    the response positions: full-vocabulary logits over a video prompt do not fit next to the VDM."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.lm_head
    ids = torch.cat([inputs["input_ids"], resp_ids[None]], dim=1)
    mask = torch.cat([inputs["attention_mask"], (resp_ids != pad_id)[None].long()], dim=1)
    kw = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    base.lm_head = torch.nn.Identity()
    try:
        hidden = model(input_ids=ids, attention_mask=mask, **kw).logits  # = final hidden states
    finally:
        base.lm_head = head
    plen = inputs["input_ids"].shape[1]
    logits = head(hidden[:, plen - 1:-1]).float()
    lp = torch.log_softmax(logits, -1).gather(-1, resp_ids[None, :, None]).squeeze(-1)[0]
    return lp, (resp_ids != pad_id).float()


def main():
    args = parse_args()
    rank, world = setup_dist()
    device = "cuda"
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)

    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "benchmark.jsonl"))]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.num_prompts]
    log(f"[rank {rank}] {len(rows)} RL prompts, world {world}")

    pipe = build_vdm(args, device, args.dit)
    agent = build_vlm(args, device, args.vlm_lora)
    clip = ClipScorer(device)
    neg = encode_text(pipe, NEGATIVE, positive=False)
    pad_id = agent.tokenizer.tokenizer.pad_token_id

    if args.stage == 1:
        pipe.dit.eval().requires_grad_(False)
        # train mode (HF gradient checkpointing only runs in train mode) with every dropout at p=0, so
        # old / new / reference log-probs of the same tokens are deterministic and comparable
        agent.train()
        for m in agent.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        agent.mllm_backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        agent.mllm_backbone.enable_input_require_grads()
        params = [p for p in agent.parameters() if p.requires_grad]
        ref_lora = {k: v.clone() for k, v in lora_state_dict(agent).items()}
    else:
        agent.eval().requires_grad_(False)
        dit = pipe.dit
        ref_dit = copy.deepcopy(dit).eval().requires_grad_(False)
        if args.vdm_lora_rank:
            from peft import LoraConfig, inject_adapter_in_model
            dit.requires_grad_(False)
            inject_adapter_in_model(LoraConfig(r=args.vdm_lora_rank, lora_alpha=args.vdm_lora_rank,
                                               target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"]), dit)
            for n_, p_ in dit.named_parameters():
                if "lora_" in n_:
                    p_.data = p_.data.float()
                    p_.requires_grad_(True)
        else:
            dit.float().requires_grad_(True)
        dit.train()
        params = [p for p in dit.parameters() if p.requires_grad]
    log(f"[rank {rank}] trainable params {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    wandb_run = None
    if rank == 0 and args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args),
                                   name=args.wandb_run_name or f"vans-grpo-s{args.stage}-{os.environ.get('SLURM_JOB_ID', 'local')}")
        except Exception as e:
            log(f"WARNING: W&B init failed ({e})")

    def save(step):
        if rank != 0:
            return
        from safetensors.torch import save_file
        d = os.path.join(args.output_dir, f"step_{step}")
        os.makedirs(d, exist_ok=True)
        if args.stage == 1:
            save_file({k: v.contiguous() for k, v in lora_state_dict(agent).items()},
                      os.path.join(d, "vlm_lora.safetensors"))
        else:
            sd = lora_state_dict(dit) if args.vdm_lora_rank else dit.state_dict()
            save_file({k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in sd.items()},
                      os.path.join(d, "dit_lora.safetensors" if args.vdm_lora_rank else "dit.safetensors"))
        log(f"saved {d}")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    ptr = rank
    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        stats = {}
        anchor = None
        while True:  # stage 2 moves on to the next prompt until one yields an anchor caption
            row = rows[ptr % len(rows)]
            ptr += world
            inputs = vlm_prompt(agent, row, args.data_dir, device)
            if args.stage == 1:
                break
            for _ in range(args.anchor_tries):
                _, texts = vlm_sample(agent, inputs, 1, 0.7, args.max_new_tokens)
                _, cap = parse_answer(texts[0])
                if rouge_l(cap, row["gt_caption"]) >= args.anchor_rouge:
                    anchor = cap
                    break
            if anchor is not None:
                break
            stats["skipped_prompts"] = stats.get("skipped_prompts", 0) + 1
        gt_frames = load_frames(os.path.join(args.data_dir, row["target_video"]))
        ref = ref_latents(encode_video(pipe, load_frames(os.path.join(args.data_dir, row["input_video"]))))

        if args.stage == 1:
            seqs, texts = vlm_sample(agent, inputs, args.group_size, args.temperature, args.max_new_tokens)
            rf, rt, rv = [], [], []
            for k, txt in enumerate(texts):
                ok, cap = parse_answer(txt)
                rf.append(ok)
                rt.append(rouge_l(cap, row["gt_caption"]))
                lat = sample_ode(pipe, pipe.dit, ref, encode_text(pipe, cap), neg, steps=args.reward_steps,
                                 cfg=args.cfg, shift=args.shift, seed=args.seed * 7919 + step * 31 + k)
                rv.append(clip.video_video(decode(pipe, lat), gt_frames))
            reward = np.array(rf) + np.array(rt) + np.array(rv)
            adv = advantages(reward)
            # old log-probs (= current policy before the update) and reference log-probs (stage-start LoRA)
            cur_lora = {k: v.clone() for k, v in lora_state_dict(agent).items()}
            with torch.no_grad():
                old_lp, ref_lp = [], []
                for s in seqs:
                    old_lp.append(response_logprobs(agent.mllm_backbone, inputs, s, pad_id)[0])
                agent.load_state_dict({k: v.to(device) for k, v in ref_lora.items()}, strict=False)
                for s in seqs:
                    ref_lp.append(response_logprobs(agent.mllm_backbone, inputs, s, pad_id)[0])
                agent.load_state_dict({k: v.to(device) for k, v in cur_lora.items()}, strict=False)
            opt.zero_grad(set_to_none=True)
            kl_sum = 0.0
            for k, s in enumerate(seqs):
                lp, m = response_logprobs(agent.mllm_backbone, inputs, s, pad_id)
                ratio = torch.exp(lp - old_lp[k])
                a = float(adv[k])
                pg = torch.maximum(-a * ratio, -a * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range))
                d = ref_lp[k] - lp
                kl = torch.exp(d) - d - 1  # k3 estimator
                loss = ((pg + args.beta * kl) * m).sum() / m.sum().clamp(min=1) / args.group_size
                loss.backward()
                kl_sum += float((kl * m).sum() / m.sum().clamp(min=1))
            allreduce_grads(params, world)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            stats = {"reward": reward.mean(), "r_format": np.mean(rf), "r_rouge": np.mean(rt), "r_clipv": np.mean(rv),
                     "kl": kl_sum / args.group_size, "grad_norm": gn.item(),
                     "resp_len": float(np.mean([(s != pad_id).sum().item() for s in seqs]))}
        else:
            ctx = encode_text(pipe, anchor)
            dit.eval()
            trajs, rv, rc = [], [], []
            for k in range(args.group_size):
                lat, traj, sig = sample_sde(pipe, dit, ref, ctx, neg, args.sde_steps, args.cfg, args.shift, args.eta,
                                            seed=args.seed * 7919 + step * 31 + k + 1000 * rank)
                frames = decode(pipe, lat)
                rv.append(clip.video_video(frames, gt_frames))
                rc.append(clip.video_text(frames, anchor))
                trajs.append(traj)
            dit.train()
            reward = np.array(rv) + np.array(rc)
            adv = advantages(reward)
            # old log-probs under the rollout policy
            with torch.no_grad():
                old = [[sde_step(x.to(device), guided_velocity(pipe, dit, x.to(device, torch.bfloat16), ref,
                                                               torch.tensor(sig[i] * 1000), ctx, neg, args.cfg).float(),
                                 sig[i], sig[i + 1], args.eta, sig[1], prev=nx.to(device))[1]
                        for x, nx, i in tr] for tr in trajs]
            n_t = len(trajs[0])
            kl_sum, clipfrac, n_terms = 0.0, 0.0, 0
            for c0 in range(0, n_t, args.timesteps_per_update):
                opt.zero_grad(set_to_none=True)
                for k, tr in enumerate(trajs):
                    for j in range(c0, min(c0 + args.timesteps_per_update, n_t)):
                        x, nx, i = tr[j]
                        x, nx = x.to(device), nx.to(device)
                        t = torch.tensor(sig[i] * 1000)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            v = guided_velocity(pipe, dit, x.to(torch.bfloat16), ref, t, ctx, neg, args.cfg,
                                                grad_ckpt=True).float()
                        _, lp = sde_step(x, v, sig[i], sig[i + 1], args.eta, sig[1], prev=nx)
                        ratio = torch.exp(lp - old[k][j])
                        a = float(adv[k])
                        pg = torch.maximum(-a * ratio, -a * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range))
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            v_ref = guided_velocity(pipe, ref_dit, x.to(torch.bfloat16), ref, t, ctx, neg,
                                                    args.cfg).float()
                        # Flow-GRPO KL between the two Gaussian steps: ||mean - mean_ref||^2 / (2 std^2)
                        s_c = min(sig[i], sig[1])
                        std2 = (args.eta ** 2) * s_c / (1 - s_c) * (sig[i] - sig[i + 1])
                        coef = (1 + (args.eta ** 2) * s_c / (1 - s_c) * (1 - sig[i]) / (2 * sig[i])) * (sig[i + 1] - sig[i])
                        kl = ((coef * (v - v_ref)) ** 2).mean() / (2 * std2)
                        loss = (pg.mean() + args.beta * kl) / (args.group_size * args.timesteps_per_update)
                        loss.backward()
                        kl_sum += kl.item()
                        clipfrac += float((ratio - 1).abs().gt(args.clip_range).float().mean())
                        n_terms += 1
                allreduce_grads(params, world)
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            stats = {**stats, "reward": reward.mean(), "r_clipv": np.mean(rv), "r_clipscore": np.mean(rc),
                     "kl": kl_sum / max(n_terms, 1), "clipfrac": clipfrac / max(n_terms, 1), "grad_norm": gn.item(),
                     "anchor_rouge": rouge_l(anchor, row["gt_caption"])}

        if args.stage == 2:
            stats.setdefault("skipped_prompts", 0)
        stats = {k: allmean(float(v), world) for k in sorted(stats) for v in [stats[k]]}
        stats["seconds_per_step"] = time.time() - t0
        if rank == 0:
            log(f"step {step}/{args.max_steps} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": step, **stats}) + "\n")
            if wandb_run:
                wandb_run.log(stats, step=step)
        if step % args.save_every == 0 or step == args.max_steps:
            save(step)
    log("GRPO COMPLETE")


if __name__ == "__main__":
    main()
