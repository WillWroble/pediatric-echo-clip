#!/bin/bash
#SBATCH --job-name=vp2_extract
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/extract_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/extract_%j.err
#SBATCH --partition=bch-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:large:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-b11-1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2

mkdir -p logs embeddings

export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
export OMP_NUM_THREADS=4

torchrun --standalone --nproc_per_node=8 extract.py \
    --avi_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/avi_manifest.csv \
    --manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/study_all.txt \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitb_bch/latest.pt \
    --model_checkpoint checkpoints/v1/latest.pt \
    --output embeddings/infonce_768_all.npz \
    --csv_dir embeddings \
    --num_segments 16 \
    --batch_size 4 \
    --num_workers 8
