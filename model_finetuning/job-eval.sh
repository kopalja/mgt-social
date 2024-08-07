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


python eval_model_on_dataframe.py --data_path "/home/kopal/multidomain.csv" --model_path ${MODEL_PATH} --base_model ${BASE_MODEL}

