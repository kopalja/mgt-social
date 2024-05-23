#!/bin/bash
set -xe

eval "$(conda shell.bash hook)"
conda activate mgt-social

declare -a models=("mdeberta-v3-base" "xlm-roberta-large" "falcon-rw-1b" "falcon-11B" "Mistral-7B-v0.1" "Meta-Llama-3-8B" "bloomz-3b" "aya-101")     

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    sbatch --output="slurm_logs/${model_name}_eval.job" -J "${model_name}"  --export=ALL,MODEL_PATH="saved_models/${model}" devana_job_eval.sh 
done
