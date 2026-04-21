#!/bin/bash
#SBATCH --job-name=vis_heat
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=logs/visualize_heatmaps_%j.out
#SBATCH --exclude=gpu-b11-1

source ~/.bashrc
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2

python -u visualize_heatmaps.py \
    --pool_npz /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v2/pool_cache/study_findings_pool_embs.npz \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining/infonce_768_all.npz \
    --checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v2/best.pt \
    --h5_path /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data/study_findings.h5 \
    --train_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/train_50_modern_nofetal.txt \
    --study_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/eval_study_sids.txt \
    --output_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/results/v2/visualize_heatmaps
