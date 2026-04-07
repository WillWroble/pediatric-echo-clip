#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/logs/pretrain_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --time=08:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env

cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining

python -u train.py \
    --chunk_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings/clip_chunks \
    --manifest ${MANIFEST:-pretrain_ids.txt} \
    --loss ${LOSS:-spectral} \
    --output ${OUTPUT:-pretrained.pt} \
    --batch_size ${BATCH_SIZE:-16384} \
    --epochs ${EPOCHS:-20} \
    --lr ${LR:-1e-3}
