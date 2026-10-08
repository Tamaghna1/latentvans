"""Shared pieces for VANS SFT and Joint-GRPO (vans_sft.py, vans_grpo.py); runs in envs/vans.

VANS released inference code and weights but not training code. Everything model-side here reuses
code/VANS (DiffSynth WanModel, VAE, model_fn_wan_video, MLLMAgent) so training sees exactly the
conditioning the released model uses at inference:
  - 6 reference latents, evenly picked from the VAE latents of 33 frames of the input video, are
    prepended to the noisy latents along time, with the same timestep, and their output is dropped;
  - text goes through umT5 (HF UMT5EncoderModel stand-in, see vans_eval_generate.HFUMT5), zero
    padded past the prompt to 512 tokens; CFG against VANS's negative prompt.
  - the VLM answers in "[Think]...[/Think][Ans]caption[/Ans]" under VANS's system prompt.

Samplers: `sample_ode` (Euler, VANS inference) and `sample_sde` (Flow-GRPO's SDE with per-step
Gaussian log-probs, Liu et al. 2025, used for VDM GRPO). Rewards follow the VANS paper section 4.
"""
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

H, W, NF = 352, 640, 33
N_REF = 6
SYSTEM_PROMPT_TAGS = re.compile(r"^\s*\[Think\](.+?)\[/Think\]\s*\[Ans\](.+?)\[/Ans\]\s*$", re.DOTALL)
NEGATIVE = ("vivid colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, "
            "stationary, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed "
            "limbs, fused fingers, stationary frame, cluttered background, three legs, crowded background "
            "people, walking backwards")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_frames(path, n=NF, h=H, w=W):
    """n evenly spaced frames, resized to cover h x w and center-cropped; list of PIL images."""
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out = []
    for i in idx:
        im = Image.fromarray(frames[i])
        s = max(h / im.height, w / im.width)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        l, t = (im.width - w) // 2, (im.height - h) // 2
        out.append(im.crop((l, t, l + w, t + h)))
    return out


# ---------------------------------------------------------------- models
def vans_imports(vans_code):
    """Put code/VANS on the path. The release imports vans.models_mllm.qwen2_5_vl, a module it does
    not ship; the class it needs is the stock one in the pinned transformers, so alias it."""
    if vans_code not in sys.path:
        sys.path.insert(0, vans_code)
    import types
    name = "vans.models_mllm.qwen2_5_vl"
    if name not in sys.modules:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen

        class Qwen2_5_VLForConditionalGeneration(_Qwen):
            """Two memory fixes to how MLLMAgent loads Qwen, neither of which changes its outputs:
            - attn_implementation "eager" -> "sdpa" (no materialized attention matrices);
            - use_cache False -> True. VANS generates with output_hidden_states=True; without a KV
              cache every new token re-runs the whole video prompt and the hidden states of all
              layers over the full sequence are kept per token, which fills a 48 GB card within a
              few hundred tokens. (Training forwards turn the cache off themselves.)"""

            @classmethod
            def from_pretrained(cls, *a, **k):
                k["attn_implementation"] = os.environ.get("VANS_QWEN_ATTN", "sdpa")
                k["use_cache"] = True
                return _Qwen.from_pretrained(*a, **k)

        shim = types.ModuleType(name)
        shim.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
        sys.modules[name] = shim


def build_vdm(args, device, dit_init):
    """VANS pipeline with VAE + umT5 + DiT. dit_init: 'base' (Wan2.1-1.3B from our diffusers
    checkpoint, converted) | path to a DiT state dict (.safetensors / .pt, e.g. VANS_vlm.safetensors)."""
    vans_imports(args.vans_code)
    from safetensors.torch import load_file
    from vans.models.model_manager import ModelManager
    from vans.models.wan_video_dit import WanModel, WanModelStateDictConverter
    from vans.pipelines.wan_video_new import WanVideoPipeline
    from vans_eval_generate import HFUMT5

    pipe = WanVideoPipeline(device=device, torch_dtype=torch.bfloat16)
    # modules WanVideoPipeline.from_pretrained would set (to None, for a T2V model); its units read them
    for name in ("dit2", "image_encoder", "motion_controller", "vace", "vace2", "audio_encoder", "audio_processor"):
        setattr(pipe, name, None)
    mm = ModelManager()
    mm.load_model(os.path.join(args.wan_orig_dir, "Wan2.1_VAE.pth"), device=device, torch_dtype=torch.bfloat16)
    pipe.vae = mm.fetch_model("wan_video_vae")
    pipe.height_division_factor = pipe.vae.upsampling_factor * 2
    pipe.width_division_factor = pipe.vae.upsampling_factor * 2
    pipe.text_encoder = HFUMT5(args.wan_diffusers_dir, device)
    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(os.path.join(args.wan_orig_dir, "google/umt5-xxl"))

    dit = WanModel(has_image_input=False, patch_size=[1, 2, 2], in_dim=16, dim=1536, ffn_dim=8960, freq_dim=256,
                   text_dim=4096, out_dim=16, num_heads=12, num_layers=30, eps=1e-6)
    if dit_init == "base":
        tdir = os.path.join(args.wan_diffusers_dir, "transformer")
        sd = {}
        for f in sorted(os.listdir(tdir)):
            if f.endswith(".safetensors"):
                sd.update(load_file(os.path.join(tdir, f)))
        sd, _ = WanModelStateDictConverter().from_diffusers(sd)
    elif dit_init.endswith(".safetensors"):
        sd = load_file(dit_init)
    else:
        sd = torch.load(dit_init, map_location="cpu")
    sd = {k[len("dit."):] if k.startswith("dit.") else k: v for k, v in sd.items() if ".mllm." not in k}
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"DiT init {dit_init}: missing {len(missing)} keys, e.g. {missing[:5]}")
    log(f"DiT from {dit_init} ({len(unexpected)} unexpected keys ignored)")
    pipe.dit = dit.to(device=device, dtype=torch.bfloat16)
    return pipe


def build_vlm(args, device, lora_init=None):
    """VANS MLLMAgent (Qwen2.5-VL-3B + LoRA r8/a32 on all linear layers, VANS's system prompt).
    lora_init: None (fresh LoRA) or a safetensors/pt with the agent's LoRA tensors."""
    vans_imports(args.vans_code)
    from vans.models_mllm.agent_mllm import MLLMAgent
    agent = MLLMAgent(mllm_pretrained_path=args.qwen_dir).to(device)
    if lora_init:
        from safetensors.torch import load_file
        sd = load_file(lora_init) if lora_init.endswith(".safetensors") else torch.load(lora_init, map_location="cpu")
        sd = {re.sub(r"^(pipe\.)?mllm\.", "", k): v for k, v in sd.items()}
        own = agent.state_dict()
        hit = {k: v for k, v in sd.items() if k in own}
        if len(hit) != len(sd):
            raise RuntimeError(f"VLM LoRA {lora_init}: {len(sd) - len(hit)} unmatched keys, e.g. "
                               f"{[k for k in sd if k not in own][:3]}")
        agent.load_state_dict(hit, strict=False)
        log(f"VLM LoRA from {lora_init} ({len(hit)} tensors)")
    return agent


def lora_state_dict(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items() if "lora_" in k}


# ---------------------------------------------------------------- VDM helpers
@torch.no_grad()
def encode_video(pipe, frames):
    """List of PIL frames -> VAE latents (1, 16, 1 + (n-1)/4, H/8, W/8) bf16."""
    v = pipe.preprocess_video(frames)
    return pipe.vae.encode(v, device=pipe.device, tiled=True).to(dtype=torch.bfloat16, device=pipe.device)


def ref_latents(full_ref):
    idx = torch.linspace(0, full_ref.shape[2] - 1, N_REF).long()
    return full_ref[:, :, idx]


@torch.no_grad()
def encode_text(pipe, text, positive=True):
    return pipe.prompter.encode_prompt(text, positive=positive, device=pipe.device).to(torch.bfloat16)


@torch.no_grad()
def decode(pipe, latents):
    v = pipe.vae.decode(latents, device=pipe.device, tiled=True)
    return pipe.vae_output_to_video(v)  # list of PIL


def velocity(pipe, dit, latents, ref, t, context, grad_ckpt=False):
    """VANS's prediction for `latents` at timestep t (scalar tensor, sigma*1000) given 6 clean ref latents."""
    from vans.pipelines.wan_video_new import model_fn_wan_video
    x = torch.cat([ref.to(latents.dtype), latents], dim=2)
    # autocast: the DiT is fp32 while trained, bf16 otherwise; inputs may be either
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model_fn_wan_video(dit=dit, latents=x, timestep=t.view(1).to(dtype=torch.bfloat16, device=latents.device),
                                 context=context, use_gradient_checkpointing=grad_ckpt)
    return out[:, :, ref.shape[2]:]


def guided_velocity(pipe, dit, latents, ref, t, ctx_pos, ctx_neg, cfg, grad_ckpt=False):
    vp = velocity(pipe, dit, latents, ref, t, ctx_pos, grad_ckpt)
    if cfg == 1.0:
        return vp
    vn = velocity(pipe, dit, latents, ref, t, ctx_neg, grad_ckpt)
    return vn + cfg * (vp - vn)


def sigmas_for(steps, shift):
    """VANS FlowMatchScheduler sigmas (sigma_min 0.003/1.002, shift), plus a final 0."""
    s = torch.linspace(1.0, 0.003 / 1.002, steps)
    s = shift * s / (1 + (shift - 1) * s)
    return torch.cat([s, torch.zeros(1)])


@torch.no_grad()
def sample_ode(pipe, dit, ref, ctx_pos, ctx_neg, steps=50, cfg=5.0, shift=5.0, seed=0, n_lat=(NF - 1) // 4 + 1):
    g = torch.Generator("cpu").manual_seed(seed)
    lat = torch.randn((1, 16, n_lat, H // 8, W // 8), generator=g).to(pipe.device, torch.bfloat16)
    sig = sigmas_for(steps, shift)
    for i in range(steps):
        v = guided_velocity(pipe, dit, lat, ref, sig[i] * 1000, ctx_pos, ctx_neg, cfg)
        lat = (lat.float() + (sig[i + 1] - sig[i]) * v.float()).to(torch.bfloat16)
    return lat


def sde_step(lat, v, s, s_next, eta, s_max_clamp, prev=None, generator=None):
    """Flow-GRPO SDE step (their eq. 9/12): returns (prev_sample, log_prob mean over dims).
    If prev is given, scores that sample instead of drawing one."""
    s_c = min(s, s_max_clamp)  # sigma=1 makes the std infinite; Flow-GRPO clamps to sigmas[1]
    std = eta * (s_c / (1 - s_c)) ** 0.5
    dt = s_next - s  # negative
    mean = lat * (1 + std ** 2 / (2 * s) * dt) + v * (1 + std ** 2 * (1 - s) / (2 * s)) * dt
    scale = std * (-dt) ** 0.5
    if prev is None:
        noise = torch.randn(lat.shape, generator=generator).to(lat.device, lat.dtype)
        prev = mean + scale * noise
    lp = -((prev.detach() - mean) ** 2) / (2 * scale ** 2) - np.log(scale) - 0.5 * np.log(2 * np.pi)
    return prev, lp.mean(dim=tuple(range(1, lp.ndim)))


@torch.no_grad()
def sample_sde(pipe, dit, ref, ctx_pos, ctx_neg, steps, cfg, shift, eta, seed, n_lat=(NF - 1) // 4 + 1):
    """SDE rollout; returns final latents and the trajectory [(x_i, x_{i+1}, i)] for GRPO training."""
    g = torch.Generator("cpu").manual_seed(seed)
    lat = torch.randn((1, 16, n_lat, H // 8, W // 8), generator=g).float().to(pipe.device)
    sig = sigmas_for(steps, shift).tolist()
    traj = []
    for i in range(steps):
        v = guided_velocity(pipe, dit, lat.to(torch.bfloat16), ref, torch.tensor(sig[i] * 1000), ctx_pos, ctx_neg,
                            cfg).float()
        if i == steps - 1:  # last step to sigma 0 is deterministic
            nxt = lat + (sig[i + 1] - sig[i]) * v
        else:
            nxt, _ = sde_step(lat, v, sig[i], sig[i + 1], eta, sig[1], generator=g)
            traj.append((lat.cpu(), nxt.cpu(), i))
        lat = nxt
    return lat.to(torch.bfloat16), traj, sig


# ---------------------------------------------------------------- rewards
class ClipScorer:
    """ViT-B/32 CLIP, as in the VANS paper's CLIP-V / CLIP-T / rewards."""

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

    @torch.no_grad()
    def images(self, frames):
        x = self.p(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device)
        return F.normalize(self.m.get_image_features(pixel_values=x), dim=-1)

    @torch.no_grad()
    def text(self, s):
        x = self.p(text=[s], return_tensors="pt", padding=True, truncation=True, max_length=77)
        return F.normalize(self.m.get_text_features(**{k: v.to(self.device) for k, v in x.items()}), dim=-1)

    def video_video(self, gen, gt):
        """Mean per-frame cosine; gen is resampled to len(gt) frames by nearest index."""
        idx = np.linspace(0, len(gen) - 1, len(gt)).round().astype(int)
        return float((self.images([gen[i] for i in idx]) * self.images(gt)).sum(-1).mean())

    def video_text(self, gen, text):
        return float((self.images(gen) @ self.text(text).T).mean())


def rouge_l(pred, ref):
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rougeL"]).score(ref, pred)["rougeL"].fmeasure


def parse_answer(text):
    """(format_ok, caption). Caption falls back to the whole text, as VANS's extract_caption does."""
    m = SYSTEM_PROMPT_TAGS.match(text)
    if m:
        return 1.0, m.group(2).strip()
    m = re.search(r"\[Ans\](.*?)\[/Ans\]", text, re.DOTALL)
    return 0.0, (m.group(1).strip() if m else text.strip())
