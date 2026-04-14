#!/bin/bash
#SBATCH --job-name=lt_gen_v1
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/logs/generate_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/logs/generate_%j.err
#SBATCH --partition=bch-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --exclude=gpu-b11-1

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer

mkdir -p logs

python -u generate.py \
    --codebook results/v3/clusters_kmeans_10k_spherical_filtered_train/codebook.npz \
    --study_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining/results/full_contrast_v6/eval/embeddings_video.npz \
    --line_embeddings results/v3/line_embeddings.npz \
    --h5_path /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data/study_findings.h5 \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_decoder/eval_manifest.txt \
    --output results/v3/heatmaps_spherical_train_v3.json \
    --split val \
    --threshold 0.6 \
    "$@"
