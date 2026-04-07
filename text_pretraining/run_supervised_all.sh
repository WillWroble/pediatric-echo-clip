#!/bin/bash
# Submit one supervised eval job per eval dir.
# Usage: bash run_supervised_all.sh
cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining
for dir in $(find results -path "*/eval/embeddings.npz" -printf "%h\n" | sort); do
    echo "Submitting: $dir"
    sbatch --job-name=eval_supervised --mem=128G --time=01:30:00 \
        --output=logs/eval_supervised_%j.out --error=logs/eval_supervised_%j.err \
        --wrap "source ~/miniconda3/etc/profile.d/conda.sh && \
                conda activate /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/env && \
                cd /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/text_pretraining && \
                python -u eval_supervised.py \
                    --embeddings ${dir}/embeddings.npz \
		    --video_embeddings /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Video_Embeddings/study_embeddings_v2.npz \
                    --h5_dir /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Line_Embeddings \
                    --labels /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/Echo_Labels_SG_Fyler_112025.csv \
                    --death_mrn /lab-share/Cardio-Mayourian-e2/Public/Echo_Clip/death_mrn.csv \
                    --output_dir ${dir}/../eval_supervised"
done
