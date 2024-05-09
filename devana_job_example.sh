#!/bin/bash
#SBATCH --account=p365-23-1  # project code
#SBATCH -J "Social posts generation"  # job name
#SBATCH --partition=gpu  # https://userdocs.nscc.sk/devana/job_submission/partitions/
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
#SBATCH --mail-type=ALL
#SBATCH --nodes=1              # Number of nodes to user
#SBATCH --gres=gpu:1           # total gpus
#SBATCH --output=job.log

# module load cuda/12.0.1

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate mgt-social


### Binary classification fine-tuning

# Demo model
# python -m aya_finetuning.finetuning --model_name google/mt5-small --demo_dataset
# python -m aya_finetuning.inference --base_model google/mt5-small



# Finetuning
# python -m aya_finetuning.finetuning --data '/home/kopal/multitude.csv'
# python -m aya_finetuning.finetuning  --demo_dataset
# python -m aya_finetuning.inference 


# Inference
# python -m aya_finetuning.inference  --data /home/kopal/multitude.csv
# python -m aya_finetuning.inference  --model_path aya_finetuning/models/merged --data /home/kopal/multitude.csv
# python -m aya_finetuning.inference  --model_path aya_finetuning/models_backup/merged --data /home/kopal/multitude.csv
# python -m aya_finetuning.inference --model_path aya_finetuning/best_binary_classification_model/merged  --data "/home/kopal/semeval_all_test.csv"
# python -m aya_finetuning.inference --model_path aya_finetuning/best_binary_classification_model/merged  --data '/home/kopal/multitude.csv'




### Instruction fine-tuning

# Finetuning
# python -m aya_finetuning.finetuning --model_name google/mt5-small --demo_dataset
python -m aya_finetuning.instruction_finetuning

