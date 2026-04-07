#!/bin/bash
#SBATCH --job-name=line_generate
##SBATCH --partition=bch-gpu-pe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder_v2/logs/generate_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder_v2/logs/generate_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_decoder_v2

python -u generate.py "$@"
