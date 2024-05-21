#!/bin/bash
set -xe

JOB_NAME=${1:-"default"}

rm -rf "lightning_logs/${JOB_NAME}"
mkdir "lightning_logs/${JOB_NAME}"

eval "$(conda shell.bash hook)"
conda activate mgt-social

declare -a models=("microsoft/mdeberta-v3-base" "FacebookAI/xlm-roberta-large" "tiiuae/falcon-rw-1b" "tiiuae/falcon-11B" "mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3-8B" "bigscience/bloomz-3b" "CohereForAI/aya-101")     

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    sbatch --output="slurm_logs/${model_name}.job" -J "${model_name}"  --export=ALL,MODEL_TYPE=${model},JOB_NAME=${JOB_NAME} devana_job.sh 
done
