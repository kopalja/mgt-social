#!/bin/bash
#SBATCH --account=p365-23-1  # project code
#SBATCH --partition=gpu  # https://userdocs.nscc.sk/devana/job_submission/partitions/
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
#SBATCH --mail-type=ALL
#SBATCH --nodes=1              # Number of nodes to user
#SBATCH --gres=gpu:1           # total gpus

# module load cuda/12.0.1
set -xe

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate mgt-social

USE_PEFT="--use_peft"
if [[ "${MODEL_TYPE}" == "FacebookAI/xlm-roberta-large" ]]; then
    USE_PEFT="--no-use_peft"
fi
# if [ "${MODEL_TYPE}" == "FacebookAI/xlm-roberta-large" ] || [ "${MODEL_TYPE}" == "microsoft/mdeberta-v3-base" ]; then
#     USE_PEFT="--no-use_peft"
# fi

echo "Running ${MODEL_TYPE}"
python main.py --data_path "/home/kopal/multidomain_subset.csv" \
               --model ${MODEL_TYPE} \
               --domain social_media \
               --language en es ru \
               --generator gemini \
               --job_name ${JOB_NAME} \
               --demo_dataset \
               ${USE_PEFT}

