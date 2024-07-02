#!/bin/bash
#SBATCH --account=p365-23-1
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
##SBATCH --time=00:10:00 # Estimate to increase job priority

## Nodes allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16
# SBATCH --ntasks-per-node=2
# SBATCH --gres=gpu:2

set -xe

# # ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate mgt-social

srun python train_gpt.py

