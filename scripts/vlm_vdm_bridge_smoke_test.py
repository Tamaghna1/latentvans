#!/usr/bin/env python
"""
VLM -> VDM bridge smoke test for LatentVANS.

Exercises the full VLM -> latent projection -> VDM bridge end to end at
batch size 1, on real (tiny, synthetic) tensors, with a real forward AND
backward pass through both models. Goal: surface shape/dtype/OOM bugs
cheaply, before writing the real Stage 0 training loop.

Pipeline under test
--------------------
1. Qwen2.5-VL-3B-Instruct (VLM) processes a synthetic video + text prompt.
   The assistant turn contains a fixed-length span of a placeholder token,
   <|latent_pad|>, bracketed by <|latent_start|> / <|latent_end|>. These are
   added to the tokenizer as new special tokens (Future-L1-style interleaved
   latent reasoning: the final-layer hidden state at each placeholder
   position IS the latent "thought").
2. Those hidden states (VLM hidden_size = 2048) are projected with a small
   trainable nn.Linear into Wan2.1-T2V-1.3B's cross-attention conditioning
   dimension (text_dim = 4096) -- this Proj module is the one genuinely new
   piece of architecture LatentVANS adds.
3. Wan2.1-T2V-1.3B (VDM, diffusers WanTransformer3DModel) is run on a real
   VAE-encoded latent of a tiny synthetic video, conditioned on the
   projected latent sequence instead of a text-encoder caption embedding.
4. A dummy diffusion loss + a dummy Stage-0-style latent-grounding loss are
   backpropagated through the whole graph (VLM -> Proj -> Wan transformer)
   and a single optimizer.step() is taken, to confirm gradients flow and
   dtypes are compatible end to end.

IMPORTANT -- checkpoint format
-------------------------------
This script uses `diffusers` classes (WanTransformer3DModel, AutoencoderKLWan)
directly, which require the DIFFUSERS-format Wan checkpoint:
    Wan-AI/Wan2.1-T2V-1.3B-Diffusers
NOT the native-format repo data_staging.sh already downloaded
(Wan-AI/Wan2.1-T2V-1.3B, no suffix -- that one has "_class_name": "WanModel"
and is meant for the official Wan2.1 GitHub repo's own code, and will NOT
load with diffusers.WanTransformer3DModel.from_pretrained).
Run download_wan_diffusers_checkpoint.sh first to fetch the diffusers-format
checkpoint as an addition (it does not touch anything data_staging.sh already
staged).

Confirmed architecture constants used below (looked up from the published
configs, not guessed):
  Qwen2.5-VL-3B-Instruct : language hidden_size = 2048
  Wan2.1-T2V-1.3B        : transformer dim = 1536 (12 heads x 128 head_dim),
                           text_dim (cross-attn conditioning) = 4096,
                           latent in/out channels = 16,
                           text encoder = google/umt5-xxl

Usage
-----
    python vlm_vdm_bridge_smoke_test.py \\
        --qwen_dir /scratch/users/anirban/tamaghnam/latentvans/checkpoints/Qwen2.5-VL-3B-Instruct \\
        --wan_diffusers_dir /scratch/users/anirban/tamaghnam/latentvans/checkpoints/Wan2.1-T2V-1.3B-Diffusers

If you hit OOM, first try --freeze_vlm (skips backward through the VLM,
keeping only the Proj module and the Wan transformer trainable -- this is
also the more realistic setup for Stage 1/2, where the VLM policy is
updated via GRPO rather than direct backprop through generation).

Running on a login node (no GPU)
---------------------------------
--device defaults to "auto": CUDA if available, CPU otherwise. On CPU,
--dtype is force-overridden to float32 (bf16/fp16 matmul support on CPU is
inconsistent across ops/torch versions) and this becomes a pure shape/dtype/
graph-wiring sanity check -- it proves the VLM->Proj->VDM plumbing is wired
correctly and catches typos/shape bugs for free, but tells you NOTHING about
real GPU memory/OOM behavior, and will be slow (minutes, not seconds) for a
3B+1.3B model pair. Use it to iterate on bugs without burning cluster queue
time, then confirm for real with sbatch/srun on a GPU node before trusting
the result. Pass --device cuda to force-fail loudly instead of silently
falling back, if you specifically meant to be on a GPU node.
"""
import argparse
import sys
import time
import traceback

import torch
import torch.nn as nn


LATENT_START = "<|latent_start|>"
LATENT_END = "<|latent_end|>"
LATENT_PAD = "<|latent_pad|>"

QWEN_HIDDEN_SIZE = 2048   # Qwen2.5-VL-3B-Instruct text hidden_size (confirmed via config.json)
WAN_TEXT_DIM = 4096       # Wan2.1-T2V-1.3B transformer text_dim / cross-attn conditioning dim


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_tensor(name, t):
    log(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
        f"requires_grad={t.requires_grad}")


def cuda_mem_report(tag):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"  [mem @ {tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qwen_dir", required=True, help="Local path to Qwen2.5-VL-3B-Instruct checkpoint")
    p.add_argument("--wan_diffusers_dir", required=True,
                   help="Local path to the DIFFUSERS-format Wan2.1-T2V-1.3B-Diffusers checkpoint "
                        "(see download_wan_diffusers_checkpoint.sh -- this is NOT the same directory "
                        "data_staging.sh downloaded)")
    p.add_argument("--latent_span_len", type=int, default=4,
                   help="Number of <|latent_pad|> tokens between latent_start/end (Future-L1's "
                        "best L_max for text answers was 4; video conditioning may want more)")
    p.add_argument("--num_video_frames", type=int, default=4, help="Frames in the synthetic VLM input video")
    p.add_argument("--frame_size", type=int, default=224, help="Height/width (square) of synthetic VLM input frames")
    p.add_argument("--gen_num_frames", type=int, default=5, help="Frames in the synthetic VDM target video (pixel space)")
    p.add_argument("--gen_height", type=int, default=192, help="Height of synthetic VDM target video (pixel space)")
    p.add_argument("--gen_width", type=int, default=192, help="Width of synthetic VDM target video (pixel space)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="'auto' (default) uses CUDA if available and silently falls back to CPU "
                        "otherwise -- useful for a quick sanity pass on a login node before spending "
                        "a GPU queue slot. 'cuda' fails loudly instead of falling back, for when you "
                        "specifically expect to be on a GPU node. CPU forces --dtype to float32 "
                        "regardless of what was passed.")
    p.add_argument("--freeze_vlm", action="store_true",
                   help="Do not backprop into the VLM -- only Proj + Wan transformer are trainable. "
                        "Use this if the full joint backward OOMs, or to match a Stage 1/2-style "
                        "setup where the VLM is updated via GRPO instead.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def dtype_from_str(s):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cuda_ok = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_ok:
        log("FATAL: --device cuda was requested but CUDA is not available here. If this is the "
            "cluster's login node, it has no GPU -- get a GPU node first "
            "(sbatch vlm_vdm_bridge_smoke_test.sh, or "
            "srun --partition=a100/h200/ada --gres=gpu:1 --pty bash), or drop --device cuda to "
            "auto-fallback to a CPU sanity check instead.")
        sys.exit(1)
    use_cuda = cuda_ok and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")

    compute_dtype = dtype_from_str(args.dtype)
    if not use_cuda and compute_dtype != torch.float32:
        log(f"device=cpu: overriding --dtype {args.dtype} -> float32 (bf16/fp16 on CPU is unreliable "
            f"across ops/torch versions; float32 is the safe choice for a wiring sanity check).")
        compute_dtype = torch.float32

    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    log(f"torch={torch.__version__} cuda_build={torch.version.cuda} "
        f"cuda_available={cuda_ok} device={device} ({gpu_name})")
    log(f"compute_dtype={compute_dtype} freeze_vlm={args.freeze_vlm} "
        f"gradient_checkpointing={args.gradient_checkpointing}")

    if not use_cuda:
        log("=" * 70)
        log("RUNNING ON CPU. This is a shape/dtype/graph-wiring sanity check ONLY -- it does NOT "
            "tell you anything about real GPU memory/OOM behavior. Qwen2.5-VL-3B + Wan2.1-1.3B will "
            "be slow on CPU (expect several minutes, not seconds). If this is the cluster's login "
            "node: keep it brief and courteous to other users, and treat a pass here as 'the plumbing "
            "is correct', not as a substitute for the real sbatch/srun GPU run. If it's too slow, "
            "shrink --num_video_frames/--frame_size/--gen_num_frames/--gen_height/--gen_width, and/or "
            "pass --freeze_vlm to cut the backward cost.")
        log("=" * 70)

    try:
        import transformers
        import diffusers
        import torchvision  # noqa: F401 -- not used directly; transformers' Qwen2VLVideoProcessor
                             # requires it as a backend and only fails on it lazily, deep inside
                             # AutoProcessor.from_pretrained, if imported this way instead.
    except ImportError as e:
        log(f"FATAL: missing dependency ({e}). Need transformers (Qwen2.5-VL support), diffusers "
            f"(Wan support), and torchvision (transformers' Qwen2-VL video processor backend -- easy "
            f"to miss since nothing else here imports it directly). Do NOT bare `pip install "
            f"--upgrade ... torchvision` to fix this -- that silently drags in a newer, unpinned "
            f"torch as torchvision's own dependency and undoes this cluster's CUDA-driver-compatible "
            f"torch pin (see claude/latentvans-handoff.md; this exact mistake caused a real "
            f"`ncclCommResume` ImportError). Instead, from vlm_vdm_bridge_smoke_test.sh's own install "
            f"lines:\n"
            f"  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121\n"
            f"  pip install transformers==5.16.1 diffusers==0.40.0 accelerate pillow numpy")
        sys.exit(1)
    log(f"transformers={transformers.__version__} diffusers={diffusers.__version__} "
        f"torchvision={torchvision.__version__}")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    # ------------------------------------------------------------------
    # 1. Load VLM + processor, add the interleaved-latent-reasoning tokens
    # ------------------------------------------------------------------
    log("=" * 70)
    log("STEP 1: loading Qwen2.5-VL-3B-Instruct")
    log("=" * 70)
    processor = AutoProcessor.from_pretrained(args.qwen_dir)
    tokenizer = processor.tokenizer
    vocab_before = len(tokenizer)

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_dir, dtype=compute_dtype, device_map=None, low_cpu_mem_usage=True,
    ).to(device)
    assert vlm.config.text_config.hidden_size == QWEN_HIDDEN_SIZE, (
        f"Expected Qwen2.5-VL-3B-Instruct hidden_size={QWEN_HIDDEN_SIZE}, "
        f"got {vlm.config.text_config.hidden_size} -- checkpoint may not be the 3B variant; "
        f"update QWEN_HIDDEN_SIZE / the Proj module input dim accordingly."
    )

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [LATENT_START, LATENT_END, LATENT_PAD]}
    )
    if num_added > 0:
        vlm.resize_token_embeddings(len(tokenizer))
    log(f"vocab size: {vocab_before} -> {len(tokenizer)} ({num_added} tokens newly added)")
    latent_pad_id = tokenizer.convert_tokens_to_ids(LATENT_PAD)

    if args.gradient_checkpointing:
        vlm.gradient_checkpointing_enable()
    if args.freeze_vlm:
        vlm.requires_grad_(False)
        log("VLM frozen (--freeze_vlm): only the resized embedding rows would need grad in real "
            "Stage 0 SFT -- here the whole VLM is frozen for a pure conditioning-path smoke test.")
    cuda_mem_report("after VLM load")

    # ------------------------------------------------------------------
    # 2. Build a synthetic (video, prompt) input with the latent span
    #    already present in the assistant turn, so we can locate its
    #    hidden states after the forward pass by token id.
    # ------------------------------------------------------------------
    log("=" * 70)
    log("STEP 2: building synthetic multimodal input")
    log("=" * 70)
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(args.seed)
    frames = [
        Image.fromarray(rng.integers(0, 255, (args.frame_size, args.frame_size, 3), dtype=np.uint8))
        for _ in range(args.num_video_frames)
    ]
    log(f"synthetic input video: {len(frames)} frames @ {args.frame_size}x{args.frame_size}")

    latent_span = LATENT_PAD * args.latent_span_len  # each occurrence tokenizes to one placeholder id
    messages = [
        {"role": "system", "content": "You are a helpful assistant that predicts what happens next in a video."},
        {"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": "What happens next in this video?"},
        ]},
        {"role": "assistant", "content": f"{LATENT_START}{latent_span}{LATENT_END} It shows the next step."},
    ]
    text = processor.apply_chat_template(messages, tokenize=False)
    inputs = processor(text=[text], videos=[frames], return_tensors="pt", padding=True).to(device)

    log_tensor("input_ids", inputs["input_ids"])
    if "pixel_values_videos" in inputs:
        log_tensor("pixel_values_videos", inputs["pixel_values_videos"])
    if "video_grid_thw" in inputs:
        log(f"  video_grid_thw: {inputs['video_grid_thw'].tolist()}")

    latent_mask = inputs["input_ids"][0] == latent_pad_id
    n_latent_positions = int(latent_mask.sum().item())
    log(f"found {n_latent_positions} <|latent_pad|> positions in input_ids "
        f"(expected {args.latent_span_len})")
    if n_latent_positions != args.latent_span_len:
        log("FATAL: latent placeholder count mismatch -- tokenizer likely merged/split the "
            "repeated <|latent_pad|> text unexpectedly. Inspect tokenizer.tokenize(latent_span).")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. VLM forward pass, extract hidden states at the latent positions,
    #    project into Wan's conditioning space.
    # ------------------------------------------------------------------
    log("=" * 70)
    log("STEP 3: VLM forward pass + latent projection")
    log("=" * 70)

    proj = nn.Linear(QWEN_HIDDEN_SIZE, WAN_TEXT_DIM, dtype=torch.float32).to(device)
    log(f"Proj module: Linear({QWEN_HIDDEN_SIZE} -> {WAN_TEXT_DIM}), params in fp32 "
        f"(cast at the input/output boundary; keeps this small head numerically stable "
        f"even though the backbones run in {compute_dtype})")

    vlm_ctx = torch.no_grad() if args.freeze_vlm else torch.enable_grad()
    with vlm_ctx:
        vlm_out = vlm(**inputs, output_hidden_states=True, return_dict=True)
    last_hidden = vlm_out.hidden_states[-1]  # (1, seq_len, 2048)
    log_tensor("vlm last_hidden_state", last_hidden)

    h_latent = last_hidden[0, latent_mask, :].unsqueeze(0)  # (1, latent_span_len, 2048)
    log_tensor("h_latent (sliced at <|latent_pad|> positions)", h_latent)

    e_latent = proj(h_latent.float()).to(compute_dtype)  # (1, latent_span_len, 4096)
    log_tensor("e_latent (projected, fed to Wan as encoder_hidden_states)", e_latent)
    cuda_mem_report("after VLM forward + projection")

    # Stage-0-style auxiliary loss: pretend we have a pooled future-frame
    # embedding target (random here -- in real Stage 0 this is
    # mean_k ViT(frame_{t+k}) from TwiFF/VANS-Data-100K). Just checks the
    # backward graph through the VLM (when not frozen) is wired correctly.
    fake_future_frame_target = torch.randn_like(h_latent)
    loss_latent_grounding = torch.nn.functional.mse_loss(h_latent.float(), fake_future_frame_target.float())
    log(f"loss_latent_grounding (dummy target) = {loss_latent_grounding.item():.4f}")

    # ------------------------------------------------------------------
    # 4. Load Wan2.1-T2V-1.3B (diffusers), VAE-encode a synthetic target
    #    video, run the transformer conditioned on e_latent.
    # ------------------------------------------------------------------
    log("=" * 70)
    log("STEP 4: loading Wan2.1-T2V-1.3B (diffusers) + VAE-encoding a synthetic video")
    log("=" * 70)

    vae = AutoencoderKLWan.from_pretrained(
        args.wan_diffusers_dir, subfolder="vae", dtype=compute_dtype, low_cpu_mem_usage=True,
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    transformer = WanTransformer3DModel.from_pretrained(
        args.wan_diffusers_dir, subfolder="transformer", dtype=compute_dtype, low_cpu_mem_usage=True,
    ).to(device)
    assert transformer.config.text_dim == WAN_TEXT_DIM, (
        f"Expected Wan transformer text_dim={WAN_TEXT_DIM}, got {transformer.config.text_dim} -- "
        f"checkpoint may not be the 1.3B variant; update WAN_TEXT_DIM / the Proj module output dim."
    )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    cuda_mem_report("after Wan VAE + transformer load")

    # Synthetic target video in pixel space, normalized to [-1, 1] like a
    # real VAE input would be.
    pixel_video = torch.rand(
        1, 3, args.gen_num_frames, args.gen_height, args.gen_width,
        device=device, dtype=compute_dtype,
    ) * 2 - 1
    log_tensor("synthetic pixel_video", pixel_video)

    with torch.no_grad():
        latent_dist = vae.encode(pixel_video).latent_dist
        video_latents = latent_dist.sample()
    log_tensor("VAE-encoded video_latents", video_latents)
    log(f"  (this is where the real in_channels={transformer.config.in_channels} check happens -- "
        f"video_latents.shape[1] must equal it)")
    assert video_latents.shape[1] == transformer.config.in_channels, (
        f"VAE output channels {video_latents.shape[1]} != transformer.in_channels "
        f"{transformer.config.in_channels}"
    )

    noise = torch.randn_like(video_latents)
    timestep = torch.randint(0, 1000, (1,), device=device).long()
    noisy_latents = video_latents + noise  # placeholder noising; swap for the real scheduler's
                                            # add_noise() once a training scheduler is wired up
    log_tensor("noisy_latents (input to transformer)", noisy_latents)
    log(f"  timestep: {timestep.item()}")

    model_pred = transformer(
        hidden_states=noisy_latents,
        timestep=timestep,
        encoder_hidden_states=e_latent,
        return_dict=False,
    )[0]
    log_tensor("model_pred (Wan transformer output)", model_pred)
    assert model_pred.shape == video_latents.shape, (
        f"model_pred shape {model_pred.shape} != video_latents shape {video_latents.shape}"
    )
    cuda_mem_report("after Wan transformer forward")

    loss_diffusion = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
    log(f"loss_diffusion (dummy target) = {loss_diffusion.item():.4f}")

    # ------------------------------------------------------------------
    # 5. Joint backward + optimizer step across the whole graph.
    # ------------------------------------------------------------------
    log("=" * 70)
    log("STEP 5: joint backward + optimizer.step()")
    log("=" * 70)

    total_loss = loss_diffusion + loss_latent_grounding
    log(f"total_loss = {total_loss.item():.4f}")

    trainable_params = list(proj.parameters()) + list(transformer.parameters())
    if not args.freeze_vlm:
        trainable_params += [p for p in vlm.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    log(f"trainable parameter count: {n_trainable:,} "
        f"({'VLM frozen' if args.freeze_vlm else 'VLM included'})")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    missing_grad = [n for n, p in proj.named_parameters() if p.grad is None]
    if missing_grad:
        log(f"FATAL: Proj module has parameters with no gradient after backward: {missing_grad} "
            f"-- the VLM->VDM bridge is disconnected from the loss graph somewhere.")
        sys.exit(1)
    log("Proj module gradients: OK (all parameters received a gradient)")

    transformer_grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1e9)
    log(f"Wan transformer grad norm: {transformer_grad_norm.item():.4f}")

    optimizer.step()
    cuda_mem_report("after backward + optimizer.step()")

    log("=" * 70)
    log("SMOKE TEST PASSED: VLM -> latent projection -> VDM forward+backward "
        "completed with no shape/dtype/OOM errors.")
    log("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        log("FATAL: CUDA OutOfMemoryError.")
        cuda_mem_report("at OOM")
        log("Try: --freeze_vlm, smaller --gen_height/--gen_width/--gen_num_frames/--num_video_frames, "
            "or --dtype bf16 if you weren't already using it.")
        sys.exit(1)
    except MemoryError:
        log("FATAL: host RAM exhausted (this is the CPU-fallback path -- see --device auto note above).")
        log("Try: --freeze_vlm, or smaller --gen_height/--gen_width/--gen_num_frames/--num_video_frames/"
            "--frame_size, or just run this on an actual GPU node instead of a login node.")
        sys.exit(1)
    except Exception:
        log("FATAL: unhandled exception.")
        traceback.print_exc()
        sys.exit(1)
