## Finetune binary classifiers

### Prerequisites
  - Installed python packages (`pip install -r requirements.txt`)
  - Enviroment with slurm job scheduler and access to gpu (tested on devana)
  - Account on Huggin face

### Finetune entrypoint
To finetune models run: `batch-job-finetune.sh <path to multidomain dataset>` 
  - Models to be finetuned are listed in batch-job-finetune.sh
  - Training parameters: `config.yaml`
  - Job logs: `slurm_logs/<model_name>.log`
  - Tensorboard logs: `lightning_logs/<job_name>`
  - Tinetuned models: `saved_models/<job_name>/<model_name>`

