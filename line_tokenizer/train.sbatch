#!/bin/bash
#SBATCH --job-name=line_tok_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/logs/train_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/logs/train_%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

python -u /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/train.py "$@"
