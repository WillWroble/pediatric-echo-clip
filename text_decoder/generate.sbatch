#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder/logs/generate_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder/logs/generate_%j.err
#SBATCH --mem=64G
#SBATCH --partition=bch-gpu-pe
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder

python -u generate.py "$@"
