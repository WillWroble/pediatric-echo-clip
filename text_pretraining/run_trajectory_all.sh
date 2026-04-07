#!/bin/bash
# Submit one trajectory analysis job per eval dir.
# Usage: bash run_trajectory_all.sh
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_vicr_pretraining

for dir in $(find results -path "*/eval/embeddings.npz" -printf "%h\n" | sort); do
    echo "Submitting: $dir"
    sbatch --job-name=traj --mem=128G --time=02:00:00 \
        --output=logs/traj_%j.out --error=logs/traj_%j.err \
        --wrap "source ~/miniconda3/etc/profile.d/conda.sh && \
                conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env && \
                cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_vicr_pretraining && \
                python -u analyze_trajectory.py \
                    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings \
                    --csv /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/echo_reports_v2.csv \
                    --eval_dir $dir \
                    --output results/trajectory_analysis.csv"
done
