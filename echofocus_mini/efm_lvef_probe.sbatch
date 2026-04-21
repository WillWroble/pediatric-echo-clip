#!/bin/bash
#SBATCH --job-name=efm_lvef_probe
#SBATCH --output=/lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini/logs/efm_infonce_%a.out
#SBATCH --error=logs/efm_infonce_%a.err
#SBATCH --array=0-3
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echofocus_mini

mkdir -p logs

FRACS=(1.0 0.5 0.25 0.1)
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}

python -u train_echofocusmini.py \
    --embeddings ${EMBEDDINGS:-../video_pretraining_v2/embeddings/jepa_70k_probe.npz} \
    --labels /lab-share/Cardio-Mayourian-e2/Public/EchoFocus_Features/Bill_Features/Subgroup_Details_from_ECGLVEF.csv \
    --train_frac $FRAC \
    --output_dir results/efm_jepa_meanpool_48_frac${FRAC} \
    --n_videos_sample 48

