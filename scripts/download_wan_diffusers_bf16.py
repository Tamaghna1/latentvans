"""Download Wan-AI/Wan2.1-T2V-1.3B-Diffusers with the big weights converted to bf16, shard by shard.

The diffusers repo stores the umT5-XXL text encoder (22.7 GB) and the DiT (5.7 GB)
in fp32. Our scratch quota can't hold 29 GB, and nothing here needs fp32 weights
(the pilot trains LoRA on a bf16 base; the text encoder only embeds captions once).
So each fp32 safetensors shard is downloaded, rewritten in bf16 next to it, and the
fp32 file deleted before the next shard -- peak extra space is one shard (~5 GB),
final size ~15 GB. Configs, tokenizer, scheduler and VAE (0.5 GB, kept fp32) are
copied as-is. The *.safetensors.index.json files keep working because shard file
names are unchanged.

Usage: python download_wan_diffusers_bf16.py --output_dir $SCRATCH/checkpoints/Wan2.1-T2V-1.3B-Diffusers-bf16
"""
import argparse
import os

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CONVERT_DIRS = ("text_encoder/", "transformer/")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--skip_text_encoder", action="store_true")
    args = p.parse_args()

    files = [f for f in HfApi().list_repo_files(REPO) if not f.startswith(("assets/", "examples/"))]
    if args.skip_text_encoder:
        files = [f for f in files if not f.startswith("text_encoder/")]
    for f in sorted(files):
        dest = os.path.join(args.output_dir, f)
        done_marker = dest + ".bf16done"
        if os.path.exists(done_marker) or (os.path.exists(dest) and not f.endswith(".safetensors")):
            continue
        if f.endswith(".safetensors") and f.startswith(CONVERT_DIRS):
            if os.path.exists(dest) and os.path.exists(done_marker):
                continue
            local = hf_hub_download(REPO, f, local_dir=args.output_dir)
            tensors = load_file(local)
            tensors = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in tensors.items()}
            tmp = local + ".tmp"
            save_file(tensors, tmp, metadata={"format": "pt"})
            del tensors
            os.replace(tmp, local)
            open(done_marker, "w").close()
            print(f"{f}: converted to bf16 ({os.path.getsize(local) / 1e9:.2f} GB)", flush=True)
        else:
            hf_hub_download(REPO, f, local_dir=args.output_dir)
            print(f"{f}: downloaded", flush=True)
    cache = os.path.join(args.output_dir, ".cache")
    print(f"done; local hub cache metadata in {cache} can be ignored", flush=True)


if __name__ == "__main__":
    main()
