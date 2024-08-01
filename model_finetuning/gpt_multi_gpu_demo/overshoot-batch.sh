#!/bin/bash
# set -xe

JOB_NAME=${1:?"Missing experiment name"}
OVERSHOOT_FACTORS=(0.1 0.5 1.0 1.5 2.0 3.0 4.0 5.0 6.0 8.0 10.0 14.0 18.0 24.0)

DST="lightning_logs/${JOB_NAME}"
if [ -d "${DST}" ]; then
    rm -rf "${DST}"
fi
mkdir -p "${DST}"

cp overshoot.py "lightning_logs/${JOB_NAME}/"
cp datasets.py "lightning_logs/${JOB_NAME}/"
cp overshoot-job.sh "lightning_logs/${JOB_NAME}/"
cp overshoot-batch.sh "lightning_logs/${JOB_NAME}/"

for factor in "${OVERSHOOT_FACTORS[@]}"; do
    sbatch --output="slurm_logs2/${JOB_NAME}___${factor}.job" -J "${JOB_NAME}"  --export=ALL,JOB_NAME=${JOB_NAME},OVERSHOOT_FACTOR=${factor} overshoot-job.sh 
done
