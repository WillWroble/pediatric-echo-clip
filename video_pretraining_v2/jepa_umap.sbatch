#!/bin/bash
#SBATCH --job-name=jepa_umap
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/jepa_umap_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/jepa_umap_%j.err
#SBATCH --partition=bch-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:large:8
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=6:00:00
#SBATCH --exclude=gpu-b11-1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2

mkdir -p logs embeddings

# Subset 2000 studies from study_val_50 (held out from aggregator training)
head -n 2000 /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/study_val_50.txt \
    > /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/umap_subset.txt

export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=8 extract.py \
    --avi_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/avi_manifest.csv \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/umap_subset.txt \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitb_bch/latest.pt \
    --model_checkpoint checkpoints/v1/latest.pt \
    --output embeddings/jepa_encode_umap_v1.npz \
    --csv_dir embeddings/umap \
    --num_segments 4 \
    --batch_size 4
