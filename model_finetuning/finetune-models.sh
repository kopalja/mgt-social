#!/bin/bash
set -xe

eval "$(conda shell.bash hook)"
conda activate mgt-social

declare -a models=("microsoft/mdeberta-v3-base" "FacebookAI/xlm-roberta-large" "tiiuae/falcon-rw-1b" "mistralai/Mistral-7B-v0.1")

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    sbatch --output="slurm_logs/${model_name}.job"  --export=ALL,MODEL_TYPE=${model} devana_job.sh 
done
