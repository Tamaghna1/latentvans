#!/bin/bash
# Builds envs/vans (VANS inference + our eval/RL code) and fetches the released VANS weights.
# VANS ships its own Qwen2.5-VL modelling code tied to a pinned transformers commit, so it
# cannot share envs/latentvans (transformers 5.x).
set -euo pipefail
T=/scratch/users/anirban/tamaghnam
export HF_HOME=$T/hf_cache PIP_CACHE_DIR=$T/hf_cache/pip
CK=$T/latentvans/checkpoints
$T/latentvans/envs/../../envs/latentvans/bin/python -m huggingface_hub.commands.huggingface_cli --help >/dev/null 2>&1 || true
P=$T/envs/latentvans/bin/python
: <<PY
from huggingface_hub import hf_hub_download
for f in ["VANS_vlm.safetensors","VANS_mllm.safetensors"]:
    print(hf_hub_download("KlingTeam/VANS", f, local_dir="$CK/VANS"))
for f in ["Wan2.1_VAE.pth","google/umt5-xxl/special_tokens_map.json","google/umt5-xxl/spiece.model","google/umt5-xxl/tokenizer.json","google/umt5-xxl/tokenizer_config.json","config.json"]:
    print(hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", f, local_dir="$CK/Wan2.1-T2V-1.3B-orig"))
PY
echo WEIGHTS_DONE
$T/miniconda3/bin/conda create -y --override-channels -c conda-forge -p $T/envs/vans python=3.12
V=$T/envs/vans/bin/pip
$V install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
$V install opencv-python-headless==4.11.0.86 diffusers==0.33.1 \
  "transformers @ git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef" \
  tokenizers==0.21.4 accelerate==1.6.0 numpy==1.26.4 decord==0.6.0 imageio==2.37.0 imageio-ffmpeg==0.6.0 \
  tqdm easydict ftfy peft safetensors einops psutil modelscope sentencepiece protobuf \
  scipy pandas rouge-score nltk wandb av
echo /scratch/users/anirban/tamaghnam/latentvans/code/VANS/vans/models_mllm/qwen-vl-utils/src > /scratch/users/anirban/tamaghnam/envs/vans/lib/python3.12/site-packages/qwen_vl_utils_local.pth
rm -rf $T/hf_cache/pip
echo ENV_DONE
