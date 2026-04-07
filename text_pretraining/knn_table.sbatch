#!/bin/bash
#SBATCH --job-name=knn_table
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/knn_table_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/logs/knn_table_%j.err
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining
python -u knn_table.py "$@"
