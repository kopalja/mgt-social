#!/bin/bash
set -xe


### Finetune models for binary classification task
# Finetuned models will be stored in:
#   saved_models/${JOB_NAME}/
# Tensorboard logs can be found in
#   lightning_logs/${JOB_NAME}/
# Slurm jobs logs are stored in
#   slurm_logs/
# WARNING: Files stored in saved_models/${JOB_NAME}/ lightning_logs/${JOB_NAME}/ and will be overwritten by this run.




DATASET=${1:?"Missing path to dataset (.csv)"}
JOB_NAME=${2:-"default"}
DOMAIN=${3:-"all"} # {'all', 'news', 'social_media'}

rm -rf "lightning_logs/${JOB_NAME}"
mkdir "lightning_logs/${JOB_NAME}"


declare -a models=("microsoft/mdeberta-v3-base" "FacebookAI/xlm-roberta-large" "tiiuae/falcon-rw-1b" "tiiuae/falcon-11B" "mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3-8B" "bigscience/bloomz-3b" "CohereForAI/aya-101")     

for model in "${models[@]}"; do
    model_name=$(basename "$model")
    sbatch --output="slurm_logs/${model_name}.job" -J "${model_name}"  --export=ALL,MODEL_TYPE=${model},JOB_NAME=${JOB_NAME},DATASET=${DATASET},DOMAIN=${DOMAIN} job-finetune.sh 
done
