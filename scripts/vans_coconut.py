"""Coconut-style continuous reasoning for the VANS VLM (Hao et al. 2024, arXiv 2412.06769), part (a):
does replacing the written [Think] reasoning with continuous thoughts give better next-event captions?

VANS's VLM answers "[Think]reasoning[/Think][Ans]caption[/Ans]". Our VANS-benchmark evaluation found
the caption quality (not the video model) is the bottleneck, so the first test is on captions alone.

Arms (all start from VANS's VLM LoRA, same data/order/steps/lr; loss = CE on the language tokens that
follow the latent part, as in Coconut):
  cot      fine-tune on the full written reasoning                              (VANS-style baseline)
  nocot    "[Think][/Think][Ans]caption[/Ans]": answer with no reasoning
  coconut  curriculum: at stage s the first s reasoning sentences are replaced by c*s continuous
           thoughts between <bot> and <eot>; at the last stage all written reasoning is removed.
           A continuous thought is the last-layer hidden state at the previous position, fed back as
           the next input embedding (one forward pass per thought; gradients flow through the chain).
  pause    same layout and curriculum as coconut, but every thought slot holds one learned <pause>
           vector (no recurrence) -- Coconut's control for "extra positions, no latent reasoning".

Input embeddings are built here (token embeddings + video features scattered at the video-token
positions) with Qwen2.5-VL's 3D rope positions from get_rope_index, so thought slots can be filled with
arbitrary vectors. <bot>, <eot> and <pause> are learned vectors; their slots hold placeholder text tokens
only so that get_rope_index assigns them ordinary sequential text positions.

  train: --mode train --arm cot|nocot|coconut|pause --data_dir data/vans_train
  eval:  --mode eval  --arm ... --ckpt <step dir or "init"> --bench_dir data/vans_eval
         greedy decoding, captions scored with BLEU@1-4 / ROUGE-L against the GT caption
"""
import argparse
import json
import os
import random
import re
import time

import torch
import torch.nn as nn

from vans_rl_common import build_vlm, log, lora_state_dict, parse_answer

SENT = re.compile(r"(?<=[.!?])\s+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--arm", choices=["cot", "nocot", "coconut", "pause"], required=True)
    p.add_argument("--data_dir")
    p.add_argument("--bench_dir")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--vans_code", required=True)
    p.add_argument("--vans_dir", required=True)
    p.add_argument("--qwen_dir", required=True)
    p.add_argument("--max_stage", type=int, default=4, help="coconut/pause: last curriculum stage")
    p.add_argument("--thoughts_per_step", type=int, default=2, help="Coconut's c")
    p.add_argument("--stage_steps", type=int, default=500, help="optimizer steps per curriculum stage")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_new", type=float, default=1e-3, help="<bot>/<eot>/<pause> vectors")
    p.add_argument("--max_answer_tokens", type=int, default=768)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--name", default=None)
    # stand-ins so build_vlm's args object has what vans_rl_common expects
    p.add_argument("--wan_orig_dir", default=None)
    p.add_argument("--wan_diffusers_dir", default=None)
    return p.parse_args()


def think_steps(think):
    return [s.strip() for s in SENT.split(think.strip()) if s.strip()]


class Coconut(nn.Module):
    def __init__(self, args, device, lora_init):
        super().__init__()
        self.agent = build_vlm(args, device, lora_init)
        bb = self.agent.mllm_backbone
        self.agent.train()
        for m in self.agent.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for n_, p_ in bb.named_parameters():
            if "lora_" in n_:
                p_.data = p_.data.float()
                p_.requires_grad_(True)
        self.base = bb.get_base_model()
        self.tok = self.agent.tokenizer.tokenizer
        hidden = self.base.config.hidden_size
        self.special = nn.Parameter(torch.randn(3, hidden, device=device) * 0.0225)  # <bot>, <eot>, <pause>
        self.placeholder = self.tok(" .", add_special_tokens=False).input_ids[-1]  # ordinary text slot
        # Qwen's last-layer hidden states have RMS ~3.9 while its input embeddings have RMS ~0.0225
        # (measured, coconut_diag job 66838). Fed back raw, a thought is a ~175x out-of-scale input and
        # gradients explode, so each thought is rescaled to the embedding RMS (direction unchanged).
        self.emb_rms = self.base.get_input_embeddings().weight.float().pow(2).mean().sqrt().item()
        self.device = device

    def lora_params(self):
        return [p for n, p in self.agent.mllm_backbone.named_parameters() if "lora_" in n]

    def text_ids(self, s):
        return self.tok(s, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.device)

    def prompt(self, video_path, instruction):
        a = self.agent
        inp, _, _ = a.tokenize(a.tokenizer, is_train=False, add_query=False, instructions=instruction, caption="",
                               video=video_path, device=self.device, torch_dtype=torch.bfloat16)
        return {k: v for k, v in inp.items() if torch.is_tensor(v) and k != "labels"}

    @torch.no_grad()
    def video_features(self, prompt):
        pv = prompt["pixel_values_videos"].to(self.base.visual.dtype)
        return self.base.visual(pv, grid_thw=prompt["video_grid_thw"])

    def build(self, prompt, vfeat, segments):
        """segments: list of ("ids", LongTensor) | ("vec", (n, hidden) tensor or None for thought slots).
        Returns (inputs_embeds, position_ids, input_ids, slot positions of each "vec" segment)."""
        ids = [prompt["input_ids"][0]]
        slots = []
        pos = ids[0].shape[0]
        for kind, x in segments:
            if kind == "ids":
                ids.append(x)
                pos += x.shape[0]
            else:
                n = x if isinstance(x, int) else x.shape[0]
                ids.append(torch.full((n,), self.placeholder, device=self.device))
                slots.append(list(range(pos, pos + n)))
                pos += n
        input_ids = torch.cat(ids)[None]
        emb = self.base.get_input_embeddings()(input_ids).float()
        vmask = (input_ids == self.base.config.video_token_id)[..., None].expand_as(emb)
        emb = emb.masked_scatter(vmask, vfeat.float())
        kw = {}
        if "second_per_grid_ts" in prompt:
            kw["second_per_grid_ts"] = prompt["second_per_grid_ts"]
        position_ids, _ = self.base.get_rope_index(input_ids, None, prompt["video_grid_thw"],
                                                   attention_mask=torch.ones_like(input_ids), **kw)
        return emb, position_ids, input_ids, slots

    def hidden(self, emb, position_ids):
        """Last-layer (normed) hidden states for a prefix of input embeddings."""
        head = self.base.lm_head
        self.base.lm_head = nn.Identity()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.agent.mllm_backbone(inputs_embeds=emb.to(torch.bfloat16), position_ids=position_ids,
                                                attention_mask=torch.ones(emb.shape[:2], device=self.device,
                                                                          dtype=torch.long)).logits
        finally:
            self.base.lm_head = head

    def logits(self, h):
        return self.base.lm_head(h.to(self.base.lm_head.weight.dtype)).float()

    def fill_thoughts(self, emb, position_ids, thought_slots, recurrent):
        """Write thoughts into emb. recurrent: thought i = hidden state at slot_i - 1 (Coconut);
        else every slot = the learned <pause> vector."""
        for p in thought_slots:
            if recurrent:
                h = self.hidden(emb[:, :p], position_ids[:, :, :p])[:, -1].float()
                h = h * (self.emb_rms / h.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt())
            else:
                h = self.special[2][None]
            emb = torch.cat([emb[:, :p], h[:, None].float(), emb[:, p + 1:]], dim=1)
        return emb

    def layout(self, arm, think, caption, stage, c, max_stage):
        """Segments after the prompt, and the index of the segment where the loss starts."""
        steps = think_steps(think)
        tail = f"[/Think][Ans]{caption}[/Ans]<|im_end|>"
        if arm == "nocot":
            return [("ids", self.text_ids("[Think]")), ("ids", self.text_ids(tail))], 1, 0
        if arm == "cot" or stage == 0:
            return [("ids", self.text_ids("[Think]")), ("ids", self.text_ids(" ".join(steps) + tail))], 1, 0
        k = min(stage, len(steps))
        n_thoughts = c * (max_stage if stage >= max_stage else k)
        rest = [] if stage >= max_stage else steps[k:]
        segs = [("ids", self.text_ids("[Think]")), ("vec", 1), ("vec", n_thoughts), ("vec", 1),
                ("ids", self.text_ids(" ".join(rest) + tail))]
        return segs, 4, n_thoughts

    def loss(self, row, data_dir, arm, stage, c, max_stage, max_answer_tokens):
        prompt = self.prompt(os.path.join(data_dir, row["input_video"]), row["instruction"])
        vfeat = self.video_features(prompt)
        segs, loss_from, n_thoughts = self.layout(arm, row["think"], row["gt_caption"], stage, c, max_stage)
        kind, last = segs[-1]
        segs[-1] = (kind, last[:max_answer_tokens])
        emb, pos, input_ids, slots = self.build(prompt, vfeat, segs)
        if slots:  # <bot>, thoughts, <eot>
            bot, thoughts, eot = slots
            emb = torch.cat([emb[:, :bot[0]], self.special[0][None, None], emb[:, bot[0] + 1:]], dim=1)
            emb = self.fill_thoughts(emb, pos, thoughts, recurrent=(arm == "coconut"))
            emb = torch.cat([emb[:, :eot[0]], self.special[1][None, None], emb[:, eot[0] + 1:]], dim=1)
        start = input_ids.shape[1] - sum(x.shape[0] for k_, x in segs[loss_from:] if k_ == "ids")
        h = self.hidden(emb, pos)
        logits = self.logits(h[:, start - 1:-1])
        target = input_ids[:, start:]
        return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)), n_thoughts

    @torch.no_grad()
    def answer(self, row, bench_dir, arm, c, max_stage, max_new_tokens):
        """Greedy answer text "[Think]...[/Think][Ans]...[/Ans]" (latent part shown as '<latent>')."""
        prompt = self.prompt(os.path.join(bench_dir, row["input_video"]), row["instruction"])
        vfeat = self.video_features(prompt)
        segs = [("ids", self.text_ids("[Think]"))]
        latent = arm in ("coconut", "pause")
        if latent:
            segs += [("vec", 1), ("vec", c * max_stage), ("vec", 1)]
        emb, pos, input_ids, slots = self.build(prompt, vfeat, segs)
        if latent:
            bot, thoughts, eot = slots
            emb[:, bot[0]] = self.special[0]
            emb = self.fill_thoughts(emb, pos, thoughts, recurrent=(arm == "coconut"))
            emb[:, eot[0]] = self.special[1]
        out = []
        end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        for _ in range(max_new_tokens):  # no KV cache: recompute the prefix (short answers, simple + exact)
            nxt = int(self.logits(self.hidden(emb, pos)[:, -1]).argmax(-1))
            if nxt == end_id:
                break
            out.append(nxt)
            ids = torch.cat([input_ids, torch.tensor([[nxt]], device=self.device)], dim=1)
            e = self.base.get_input_embeddings()(ids[:, -1:]).float()
            emb = torch.cat([emb, e], dim=1)
            kw = {"second_per_grid_ts": prompt["second_per_grid_ts"]} if "second_per_grid_ts" in prompt else {}
            pos, _ = self.base.get_rope_index(ids, None, prompt["video_grid_thw"],
                                              attention_mask=torch.ones_like(ids), **kw)
            input_ids = ids
        text = self.tok.decode(out, skip_special_tokens=True)
        return "[Think]" + ("<latent>" if latent else "") + text


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "benchmark.jsonl"))]
    model = Coconut(args, device, os.path.join(args.vans_dir, "VANS_mllm.safetensors"))
    curriculum = args.arm in ("coconut", "pause")
    n_stages = args.max_stage + 1 if curriculum else 1
    total = args.stage_steps * (args.max_stage + 1)  # same optimizer steps for every arm
    log(f"arm {args.arm}: {len(rows)} rows, {total} optimizer steps x {args.grad_accum} accum, "
        f"{n_stages} stage(s)")

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args),
                                   name=args.wandb_run_name or f"coconut-{args.arm}-{os.environ.get('SLURM_JOB_ID', 'local')}")
        except Exception as e:
            log(f"WARNING: W&B init failed ({e})")

    def new_opt():  # Coconut resets the optimizer at each stage
        return torch.optim.AdamW([{"params": model.lora_params(), "lr": args.lr},
                                  {"params": [model.special], "lr": args.lr_new}], weight_decay=0.01)

    def save(tag):
        from safetensors.torch import save_file
        d = os.path.join(args.output_dir, tag)
        os.makedirs(d, exist_ok=True)
        save_file({k: v.contiguous() for k, v in lora_state_dict(model.agent).items()},
                  os.path.join(d, "vlm_lora.safetensors"))
        torch.save({"special": model.special.detach().cpu()}, os.path.join(d, "special.pt"))
        log(f"saved {d}")

    rng = random.Random(args.seed)  # same row order for every arm
    order, pos, run, fails, t0 = list(range(len(rows))), len(rows), 0.0, 0, time.time()
    opt, stage = new_opt(), -1
    params = model.lora_params() + [model.special]
    for step in range(1, total + 1):
        s = min((step - 1) // args.stage_steps, args.max_stage) if curriculum else 0
        if s != stage:
            if stage >= 0 and curriculum:
                save(f"stage_{stage}")
                opt = new_opt()
            stage = s
            log(f"stage {stage}")
        opt.zero_grad(set_to_none=True)
        nt = 0
        for _ in range(args.grad_accum):
            if pos >= len(rows):
                rng.shuffle(order)
                pos = 0
            r = rows[order[pos]]
            pos += 1
            try:
                loss, nt = model.loss(r, args.data_dir, args.arm, stage, args.thoughts_per_step, args.max_stage,
                                      args.max_answer_tokens)
                (loss / args.grad_accum).backward()
                run += loss.item() / args.grad_accum
                fails = 0
            except Exception as e:
                fails += 1
                log(f"WARNING: {r['sample_id']}: {str(e)[-300:]}")
                if fails >= 20:
                    raise
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.log_every == 0:
            dt = (time.time() - t0) / args.log_every
            row = {"loss": run / args.log_every, "grad_norm": gn.item(), "stage": stage, "n_thoughts": nt,
                   "seconds_per_step": dt}
            log(f"step {step}/{total} " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in row.items()))
            with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, **row}) + "\n")
            if wandb_run:
                wandb_run.log(row, step=step)
            run, t0 = 0.0, time.time()
    save("final")
    log("TRAINING COMPLETE")


def evaluate(args):
    device = "cuda"
    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))]
    rows = (rows[:args.limit] if args.limit else rows)[args.shard::args.num_shards]
    init = os.path.join(args.vans_dir, "VANS_mllm.safetensors")
    model = Coconut(args, device, init if args.ckpt == "init" else os.path.join(args.ckpt, "vlm_lora.safetensors"))
    if args.ckpt != "init":
        model.special.data = torch.load(os.path.join(args.ckpt, "special.pt"))["special"].to(device)
    model.eval()
    name = args.name or f"coconut_{args.arm}"
    out = os.path.join(args.output_dir, name)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"captions_shard{args.shard}.jsonl")
    done = {json.loads(l)["sample_id"] for l in open(path)} if os.path.exists(path) else set()
    for i, r in enumerate(rows):
        if r["sample_id"] in done:
            continue
        t0 = time.time()
        raw = model.answer(r, args.bench_dir, args.arm, args.thoughts_per_step, args.max_stage, args.max_new_tokens)
        ok, cap = parse_answer(raw.replace("<latent>", ""))
        with open(path, "a") as f:
            f.write(json.dumps({"sample_id": r["sample_id"], "caption": cap, "raw": raw, "format_ok": ok}) + "\n")
        log(f"[{name} shard {args.shard}] {i + 1}/{len(rows)} ({time.time() - t0:.0f}s) {cap[:80]!r}")
    log("EVAL GENERATION DONE")


if __name__ == "__main__":
    a = parse_args()
    train(a) if a.mode == "train" else evaluate(a)
