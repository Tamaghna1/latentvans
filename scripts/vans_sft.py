"""VANS SFT, reimplemented from the paper (arXiv 2511.16669 section 5) on top of code/VANS.

  --stage vlm  Qwen2.5-VL-3B LoRA (r8, alpha 32, all linear layers) trained to answer
               "[Think]<ENG_Think>[/Think][Ans]<ENG_GT_Caption>[/Ans]" for (input video, instruction),
               CE on the answer tokens only. Paper: 10K steps, lr 5e-5.
  --stage vdm  Wan2.1-1.3B DiT, all blocks trained, on the target clip (33 frames, 352x640) given the
               GT caption and 6 reference latents of the input clip, with VANS's own training loss
               (FlowMatchScheduler training timesteps, shift 5, timestep weighting). Paper: 20K steps, lr 5e-5.

Data: build_vans_eval.py --split train output (benchmark.jsonl format). Checkpoints hold only
trained tensors: VLM LoRA (vlm_lora.safetensors) or the full DiT (dit.safetensors), which
vans_grpo.py / vans_eval_generate.py load.
"""
import argparse
import json
import os
import random
import time

import torch

from vans_rl_common import (build_vdm, build_vlm, encode_text, encode_video, load_frames, log,
                            lora_state_dict, ref_latents, velocity)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["vlm", "vdm"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vans_code", required=True)
    p.add_argument("--wan_orig_dir", required=True)
    p.add_argument("--wan_diffusers_dir", required=True)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache_dir", default=None,
                   help="--stage vdm: cache each row's ref/target latents + caption embedding here on first use")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)
    device = "cuda"
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "benchmark.jsonl"))]
    log(f"{len(rows)} training rows")

    if args.stage == "vlm":
        agent = build_vlm(args, device)
        agent.train()
        params = [p for p in agent.parameters() if p.requires_grad]
        tok = agent.tokenizer

        def step_loss(r):
            vid = os.path.join(args.data_dir, r["input_video"])
            prompt, _, _ = agent.tokenize(tok, is_train=False, add_query=False, instructions=r["instruction"],
                                          caption=r["gt_caption"], video=vid, device=device,
                                          torch_dtype=torch.bfloat16)
            full, _, _ = agent.tokenize(tok, is_train=False, add_query=True, instructions=r["instruction"],
                                        caption=r["gt_caption"], thinking=r["think"], video=vid, device=device,
                                        torch_dtype=torch.bfloat16)
            full = {k: v for k, v in full.items() if k != "labels"}  # VANS's tokenize adds its own; we mask ourselves
            labels = full["input_ids"].clone()
            labels[:, :prompt["input_ids"].shape[1]] = -100
            out = agent.mllm_backbone(**full, labels=labels)
            return out.loss
    else:
        pipe = build_vdm(args, device, "base")
        dit = pipe.dit
        dit.float().train()
        dit.requires_grad_(True)
        params = list(dit.parameters())
        sch = pipe.scheduler
        sch.set_timesteps(1000, training=True, shift=5.0)

        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)

        def encode_row(r):
            path = os.path.join(args.cache_dir, f"{r['sample_id']}.pt") if args.cache_dir else None
            if path and os.path.exists(path):
                c = torch.load(path, map_location=device)
            else:
                c = {"ref": ref_latents(encode_video(pipe, load_frames(os.path.join(args.data_dir, r["input_video"])))),
                     "x0": encode_video(pipe, load_frames(os.path.join(args.data_dir, r["target_video"]))),
                     "ctx": encode_text(pipe, r["gt_caption"])}
                if path:
                    n = int((c["ctx"][0].abs().sum(-1) > 0).sum())  # store unpadded; re-pad on load
                    torch.save({"ref": c["ref"].cpu(), "x0": c["x0"].cpu(), "ctx": c["ctx"][:, :n].cpu(),
                                "ctx_len": c["ctx"].shape[1]}, path)
                    return c
            if "ctx_len" in c:
                c["ctx"] = torch.nn.functional.pad(c["ctx"], (0, 0, 0, c["ctx_len"] - c["ctx"].shape[1]))
            return c

        def step_loss(r):
            c = encode_row(r)
            ref, x0, ctx = c["ref"], c["x0"].float(), c["ctx"]
            tid = torch.randint(0, len(sch.timesteps), (1,))
            t = sch.timesteps[tid].to(device)
            noise = torch.randn_like(x0)
            xt = sch.add_noise(x0, noise, t)
            target = sch.training_target(x0, noise, t)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = velocity(pipe, dit, xt.to(torch.bfloat16), ref, t, ctx, grad_ckpt=True)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            return loss * sch.training_weight(t).to(device)

    log(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args),
                                   name=args.wandb_run_name or f"vans-sft-{args.stage}-{os.environ.get('SLURM_JOB_ID', 'local')}")
        except Exception as e:
            log(f"WARNING: W&B init failed ({e})")

    def save(step):
        d = os.path.join(args.output_dir, f"step_{step}")
        os.makedirs(d, exist_ok=True)
        from safetensors.torch import save_file
        if args.stage == "vlm":
            save_file(lora_state_dict(agent), os.path.join(d, "vlm_lora.safetensors"))
        else:
            save_file({k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in dit.state_dict().items()},
                      os.path.join(d, "dit.safetensors"))
        log(f"saved {d}")

    order, pos, t0, run = list(range(len(rows))), len(rows), time.time(), 0.0
    consecutive_failures = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            if pos >= len(rows):
                random.shuffle(order)
                pos = 0
            r = rows[order[pos]]
            pos += 1
            try:
                loss = step_loss(r) / args.grad_accum
            except Exception as e:  # unreadable clip etc.: skip the micro-batch, keep training
                consecutive_failures += 1
                log(f"WARNING: {r['sample_id']}: {str(e)[-300:]}")
                if consecutive_failures >= 20:
                    raise RuntimeError("20 consecutive training samples failed; this is a bug, not bad data") from e
                continue
            consecutive_failures = 0
            loss.backward()
            run += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0:
            dt = (time.time() - t0) / args.log_every
            log(f"step {step}/{args.max_steps} loss={run / args.log_every:.4f} grad_norm={gn.item():.3f} ({dt:.2f}s/step)")
            if wandb_run:
                wandb_run.log({"loss": run / args.log_every, "grad_norm": gn.item(), "lr": sched.get_last_lr()[0],
                               "seconds_per_step": dt}, step=step)
            t0, run = time.time(), 0.0
        if step % args.save_every == 0 or step == args.max_steps:
            save(step)
    log("SFT COMPLETE")


if __name__ == "__main__":
    main()
