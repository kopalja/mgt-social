#!/bin/bash
#SBATCH --account=p365-23-1
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
##SBATCH --time=hh:mm:ss # Estimate to increase job priority

## Nodes allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2

set -xe

module load CUDA/12.1.1

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate mgt-social

USE_PEFT="--use_peft"
if [[ "${MODEL_TYPE}" == "FacebookAI/xlm-roberta-large" ]]; then
    USE_PEFT="--no-use_peft"
fi

USE_PEFT="--no-use_peft" # TODO: remove

echo "Running ${MODEL_TYPE}"
# accelerate launch main.py --data_path ${DATASET} \
srun python main.py --data_path ${DATASET} \
                    --model ${MODEL_TYPE} \
                    --domain ${DOMAIN} \
                    --job_name ${JOB_NAME} \
                    ${USE_PEFT}

