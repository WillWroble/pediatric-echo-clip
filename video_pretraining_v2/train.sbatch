#!/bin/bash
#SBATCH --job-name=vp2_train
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/train_%j.out
#SBATCH --error=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2/logs/train_%j.err
##SBATCH --partition=bch-gpu
#SBATCH --partition=bch-gpu-pe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:large:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=750G
#SBATCH --time=1-00:00:00
##SBATCH --exclude=gpu-b11-1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/video_pretraining_v2

mkdir -p logs checkpoints

export PYTHONPATH=/lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA:$PYTHONPATH
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc_per_node=8 train.py \
    --avi_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/avi_manifest.csv \
    --pretrain_manifest /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/manifests/pretrain.txt \
    --encoder_checkpoint /lab-share/Cardio-Mayourian-e2/Public/Echo_JEPA/checkpoints/echojepa_vitb_bch/latest.pt \
    --output_dir checkpoints/v2 \
    --m_studies 2048 \
    --k_clips 2 \
    --epochs 20 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --warmup_steps 500 \
    --temperature 0.1 \
    --num_workers 2 \
    --intra_mrn
    #--resume checkpoints/v1/latest.pt
