#!/bin/bash
#SBATCH --job-name=eval_supervised_v2
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/eval_supervised_v2_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/eval_supervised_v2_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining
python -u eval_supervised_v2.py "$@"



