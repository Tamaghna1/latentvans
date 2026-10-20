"""Scene-anchored answer videos without training (plan.md P28 step 1): make the video model continue the input
scene instead of redrawing it. Same model, captions and settings as vans_eval_generate.py (VANS DiT, 6 reference
latents from the input clip, 352x640, 33 frames, 50 steps, CFG 5, seed 0); only the anchoring differs:

  --ffrep        first-frame replacement: after every denoising step, the first generated latent frame is reset
                 to the VAE latent of the last input frame, noised to the step's level ((1 - s) z + s eps, flow
                 matching), so it ends exactly on that frame and the model fills in the rest around it. Latent
                 frame 0 of the causal Wan VAE decodes to video frame 0 alone.
  --strength S   SDEdit: start from the frozen last frame (33 copies) noised to level S instead of pure noise
                 (S = denoising_strength before the scheduler's shift 5: 0.3 -> sigma 0.68, 0.5 -> 0.83).
Both can be combined. Captions: --captions glob of jsonl (sample_id, caption), or 'gt' for the benchmark's.
--every K keeps every K-th benchmark row (screening on a subset; rows alternate sources at random, so a subset
mixes COIN and YouCook).

Output: <out_dir>/<name>/{videos/<sample_id>.mp4, captions_shard<k>.jsonl}, the layout vans_eval_metrics.py and
event_judge.py read; resumable, shardable.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from vans_eval_generate import F, H, NEGATIVE, W, build_pipe, load_frames, log  # noqa: E402


class FirstFrameAnchor:
    """Wraps scheduler.step: after each step, the first of the last n_gen latent frames (the generated part; the
    pipeline prepends its reference latents during the step) becomes the anchor latent at the next noise level."""

    def __init__(self, scheduler, n_gen):
        self.sch, self.n_gen, self.orig = scheduler, n_gen, scheduler.step
        self.z = self.eps = None
        scheduler.step = self.step

    def set(self, z, seed=0):
        self.z = z
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.eps = torch.randn(z.shape, generator=g).to(z.device, z.dtype)

    def step(self, model_output, timestep, sample, to_final=False, **kw):
        out = self.orig(model_output, timestep, sample, to_final=to_final, **kw)
        if self.z is not None:
            i = torch.argmin((self.sch.timesteps - timestep.to(self.sch.timesteps.device)).abs()).item()
            s = 0.0 if to_final or i + 1 >= len(self.sch.timesteps) else float(self.sch.sigmas[i + 1])
            k = out.shape[2] - self.n_gen
            out[:, :, k:k + 1] = ((1 - s) * self.z + s * self.eps).to(out.dtype)
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--captions", required=True, help="glob of caption jsonl files, or 'gt'")
    p.add_argument("--ffrep", action="store_true")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vans_code", required=True)
    p.add_argument("--vans_dir", required=True)
    p.add_argument("--wan_orig_dir", required=True)
    p.add_argument("--wan_diffusers_dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if not args.ffrep and args.strength >= 1.0:
        raise SystemExit("no anchoring requested: use --ffrep and/or --strength < 1")

    from vans_rl_common import vans_imports
    vans_imports(args.vans_code)
    from vans import save_video
    rows = [json.loads(l) for l in open(os.path.join(args.bench_dir, "benchmark.jsonl"))][::args.every]
    rows = rows[:args.limit] if args.limit else rows
    rows = rows[args.shard::args.num_shards]
    if args.captions == "gt":
        given = {r["sample_id"]: r["gt_caption"] for r in rows}
    else:
        given = {}
        for f in sorted(glob.glob(args.captions)):
            for l in open(f):
                c = json.loads(l)
                given[c["sample_id"]] = c["caption"]
        if not given:
            raise RuntimeError(f"no captions match {args.captions}")
    out = os.path.join(args.out_dir, args.name)
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    cap_path = os.path.join(out, f"captions_shard{args.shard}.jsonl")
    done = {json.loads(l)["sample_id"] for l in open(cap_path)} if os.path.exists(cap_path) else set()

    args.qwen_dir = None
    pipe = build_pipe(args, use_mllm=False)
    n_gen = (F - 1) // 4 + 1  # latent frames of the generated clip (causal VAE, 4x temporal)
    anchor = FirstFrameAnchor(pipe.scheduler, n_gen) if args.ffrep else None
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        if sid in done:
            continue
        caption = given.get(sid)
        if caption is None:
            log(f"WARNING: no caption for {sid}, skipped")
            continue
        t0 = time.time()
        ref = load_frames(os.path.join(args.bench_dir, r["input_video"]))
        last = ref[-1]  # last input frame, covered and cropped to H x W like the reference frames
        if anchor:
            with torch.no_grad():
                z = pipe.vae.encode(pipe.preprocess_video([last]), device=pipe.device, tiled=True)
            anchor.set(z.to(device=pipe.device, dtype=pipe.torch_dtype))
            if i == 0:
                log(f"anchor latent {tuple(z.shape)}, generated latent frames {n_gen}")
        kw = dict(input_video=[last] * F, denoising_strength=args.strength) if args.strength < 1 else {}
        video = pipe(prompt=caption, negative_prompt=NEGATIVE, seed=0, tiled=True, height=H, width=W, num_frames=F,
                     ref_VAE_video=ref, **kw)
        save_video(video, os.path.join(out, "videos", f"{sid}.mp4"), fps=11, quality=5)
        with open(cap_path, "a") as f:
            f.write(json.dumps({"sample_id": sid, "caption": caption}) + "\n")
        log(f"[{args.name} shard {args.shard}] {i + 1}/{len(rows)} {sid} ({time.time() - t0:.0f}s)")
    log("GENERATION DONE")


if __name__ == "__main__":
    main()
