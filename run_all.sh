#!/bin/bash
set -xe

# declare -a models=("vicuna" "mistral")
declare -a models=("mistral" "vicuna")
declare -a methods=("k_to_one" "paraphrase")

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        echo "Processing: model: ${model},  method: ${method}"
        python main.py --model_name ${model} --type ${method}
    done
done

