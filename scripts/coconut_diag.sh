#!/bin/bash
#SBATCH --job-name=coconut_diag
#SBATCH --partition=ada
#SBATCH --output=/scratch/users/anirban/tamaghnam/latentvans/logs/coconut_diag_%j.out
#SBATCH --error=/scratch/users/anirban/tamaghnam/latentvans/logs/coconut_diag_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
T=/scratch/users/anirban/tamaghnam; S=$T/latentvans
export HF_HOME=$T/hf_cache PYTHONPATH=$S/scripts
$T/envs/vans/bin/python $S/scripts/coconut_diag.py --vans_code $S/code/VANS --vans_dir $S/checkpoints/VANS --qwen_dir $S/checkpoints/Qwen2.5-VL-3B-Instruct
