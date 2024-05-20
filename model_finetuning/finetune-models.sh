#!/bin/bash
set -xe

eval "$(conda shell.bash hook)"
conda activate mgt-social

declare -a models=("microsoft/mdeberta-v3-base" "FacebookAI/xlm-roberta-large" "tiiuae/falcon-rw-1b" "tiiuae/falcon-11B" "mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3-8B" "bigscience/bloomz-3b" "google/mt5-small")

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    sbatch --output="slurm_logs/${model_name}.job" -J "${model_name}"  --export=ALL,MODEL_TYPE=${model} devana_job.sh 
done
