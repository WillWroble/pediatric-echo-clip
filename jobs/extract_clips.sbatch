#!/bin/bash
#SBATCH --job-name=extract_clips
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/extract_%A_%a.out
#SBATCH --array=0-39
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

DATA_DIR=/lab-share/Cardio-Mayourian-e2/Public/Echo_Embeddings/Embeddings/Internal
CHUNK_DIR=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings

python -u /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/code/extract_clips.py $DATA_DIR $CHUNK_DIR

