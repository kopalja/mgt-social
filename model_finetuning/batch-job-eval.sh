#!/bin/bash
set -xe

JOB_NAME=${1:?"Missing job name"}

declare -a models=("microsoft/mdeberta-v3-base" "FacebookAI/xlm-roberta-large" "tiiuae/falcon-rw-1b" "tiiuae/falcon-11B" "mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3-8B" "bigscience/bloomz-3b" "CohereForAI/aya-101")     

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    sbatch --output="slurm_logs/${JOB_NAME}_${model_name}_eval.job" -J "eval_${JOB_NAME}_${model_name}"  --export=ALL,BASE_MODEL=${model},MODEL_PATH="saved_models/${JOB_NAME}/${model_name}" job-eval.sh 
done
