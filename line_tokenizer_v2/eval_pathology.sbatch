#!/bin/bash
#SBATCH --job-name=eval_path
##SBATCH --partition=bch-gpu-pe
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_pathology_%j.out
#SBATCH --exclude=gpu-b11-1

source ~/.bashrc
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2

python -u eval_pathology.py \
    --checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v2/best.pt \
    --anchors anchor_pathology_lines.txt \
    --h5_path /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data/study_findings.h5 \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining/infonce_768_all.npz \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/val_50_modern_nofetal.txt \
    --max_videos 128 \
    --output_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v2/eval_pathology
