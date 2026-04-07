#!/bin/bash
#SBATCH --job-name=efm_embed
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/efm_embed_%j.out
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini

python -u embed.py \
    --checkpoint ${CHECKPOINT:-results/best.pt} \
    --embeddings ${EMBEDDINGS:-infonce_768.npz} \
    --output ${OUTPUT:-/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings/study_embeddings_v2.npz}
