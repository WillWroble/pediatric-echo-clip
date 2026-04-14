#!/bin/bash
#SBATCH --job-name=lt_gen_v2
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/generate_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2/logs/generate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer_v2

mkdir -p logs

python -u generate.py \
    --codebook results/v1/clusters_spherical_kmeans_10k_modern/codebook.npz \
    --checkpoint results/v1/latest.pt \
    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining/infonce_768_all.npz \
    --h5_path /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/data/study_findings.h5 \
    --line_embeddings results/v1/line_embeddings.npz \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_decoder/eval_manifest.txt \
    --output results/v1/heatmaps_spherical_modern_v4.json \
    --line_filters /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/line_tokenizer/ignore_patterns.txt \
    --threshold 0.65 \
    --linkage_cutoff 0.8 \
    "$@"
