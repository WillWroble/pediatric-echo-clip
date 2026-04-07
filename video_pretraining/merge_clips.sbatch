#!/bin/bash
#SBATCH --job-name=merge_clips
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/merge_clips_%j.out
#SBATCH --mem=512G
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

CHUNK_DIR=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings
OUT_PATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings/clip_embeddings.npz

python -u /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining/merge_clip_embeddings.py \
    $CHUNK_DIR $OUT_PATH

