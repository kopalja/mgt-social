#!/bin/bash
set -xe

# declare -a models=("gemini" "eagle" "vicuna" "mistral")
declare -a models=("aya")
declare -a methods=("keywords" "k_to_one" "paraphrase")
# declare -a methods=("paraphrase")

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        python main.py --model_name ${model} --type ${method} --cache_dir "/mnt/jakub.kopal/"
    done
done

