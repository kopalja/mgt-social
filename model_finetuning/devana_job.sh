#!/bin/bash
#SBATCH --account=p365-23-1  # project code
#SBATCH -J "Social posts generation"  # job name
#SBATCH --partition=gpu  # https://userdocs.nscc.sk/devana/job_submission/partitions/
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
#SBATCH --mail-type=ALL
#SBATCH --nodes=1              # Number of nodes to user
#SBATCH --gres=gpu:1           # total gpus

# module load cuda/12.0.1

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate mgt-social


echo "Running ${MODEL_TYPE}"
python main.py --data_path "/home/kopal/multidomain_subset.csv" \
               --model ${MODEL_TYPE} \
               --domain social_media \
               --language en es ru \
               --generator gemini \
               --hf_token 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

