#!/bin/bash
#SBATCH --job-name=gpu_probe
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/gpu_probe_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/gpu_probe_%j.out
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Which GPU does a job see, and can each env use it? (diagnosing CUDA failures on some nodes)
hostname; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,driver_version,compute_mode,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
for e in vans latentvans; do /scratch/users/anirban/tamaghnam/envs/$e/bin/python -c "import torch;print(\"$e\",torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.device_count())" 2>&1 | tail -1; done
