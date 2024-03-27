#!/bin/bash
set -xe

declare -a models=("gemini" "eagle" "vicuna" "mistral")
declare -a methods=("keywords" "k_to_one" "paraphrase")

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        python main.py --model_name ${model} --type ${method}
    done
done

