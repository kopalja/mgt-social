#!/bin/bash
#SBATCH --account=p365-23-1  # project code
#SBATCH --partition=gpu  # https://userdocs.nscc.sk/devana/job_submission/partitions/
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
#SBATCH --mail-type=ALL
#SBATCH --nodes=1              # Number of nodes to user
#SBATCH --gres=gpu:1           # total gpus

set -xe

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate mgt-social

USE_PEFT="--use_peft"
if [[ "${MODEL_TYPE}" == "FacebookAI/xlm-roberta-large" ]]; then
    USE_PEFT="--no-use_peft"
fi

echo "Running ${MODEL_TYPE}"
python main.py --data_path ${DATASET} \
               --model ${MODEL_TYPE} \
               --domain ${DOMAIN} \
               --job_name ${JOB_NAME} \
               ${USE_PEFT}

