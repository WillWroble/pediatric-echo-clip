#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/embed_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=04:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining

python -u embed.py \
    --checkpoint ${CHECKPOINT:-pretrained.pt} \
    --chunk_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings/clip_chunks \
    --manifest ${MANIFEST:-downstream_ids.txt} \
    --output ${OUTPUT:-embeddings.npz}
