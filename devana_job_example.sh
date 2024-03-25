#!/bin/bash
#SBATCH --account=p365-23-1  # project code
#SBATCH -J "test-mgt"  # job name
#SBATCH --partition=gpu  # https://userdocs.nscc.sk/devana/job_submission/partitions/
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
#SBATCH --mail-type=ALL
#SBATCH --nodes=1              # Number of nodes to user
#SBATCH --time=0-00:10:00      # 10 minutes of wall time
#SBATCH --gres=gpu:2           # total gpus
#SBATCH --output=job.log

# module load cuda/12.0.1

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate mgt-social
./run_all.sh
