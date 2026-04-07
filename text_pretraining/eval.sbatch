#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/eval_vicreg_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/eval_vicreg_%j.err
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining

python -u eval.py "$@"
