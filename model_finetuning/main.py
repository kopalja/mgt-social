import argparse
import os
import pandas as pd
import pytorch_lightning as pl


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModelForSequenceClassification
from huggingface_hub import login

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
)

from misc import QUANTIZATION_CONFIG

from model_trainer import TrainerForSequenceClassification

if __name__ == "__main__":
    ### Example
    # python main.py --data_path "/home/kopal/multidomain.csv" --model microsoft/mdeberta-v3-base --domain social_media --language en es ru --generator gemini
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--model",
        choices=[
            "microsoft/mdeberta-v3-base",
            "FacebookAI/xlm-roberta-large",
            "tiiuae/falcon-rw-1b",
            "tiiuae/falcon-11B",
            "mistralai/Mistral-7B-v0.1",
            "meta-llama/Meta-Llama-3-8B",
            "bigscience/bloomz-3b",
            "google/mt5-small",
            "CohereForAI/aya-101",
        ],
        nargs="?",
        required=True,
    )
    parser.add_argument("--domain", choices=["news", "social_media"], nargs="?")
    parser.add_argument(
        "--language",
        choices=["en", "pt", "de", "nl", "es", "ru", "pl", "ar", "bg", "ca", "uk", "pl", "ro"],
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--generator",
        choices=[
            "gemini",
            "gpt-3.5-turbo-0125",
            "opt-iml-max-30b",
            "aya-101",
            "v5-Eagle-7B-HF",
            "Mistral-7B-Instruct-v0.2",
            "vicuna-13b",
            "Llama-2-70b-chat-hf",
        ],
        nargs="?",
    )
    parser.add_argument("--hf_token", type=str)
    parser.add_argument('--job_name', type=str, default="default")
    parser.add_argument('--use_peft', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--demo_dataset', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)


    # 1) Create datatset
    df = pd.read_csv(args.data_path, index_col=0)
    if args.language:
        df = df[df['language'].isin(args.language)]
    if args.generator:
        df = df[df['multi_label'].isin([args.generator, "human"])]
    if args.domain:
        df = df[df['domain'] == args.domain]

    
    target_map = {
        'microsoft/mdeberta-v3-base': ['query_proj', 'key_proj', 'value_proj'],
        'FacebookAI/xlm-roberta-large': ['query', 'key', 'value'],
        'tiiuae/falcon-rw-1b': ['query_key_value'],
        'tiiuae/falcon-11B': ['query_key_value'],
        'mistralai/Mistral-7B-v0.1': ['q_proj', 'k_proj', 'v_proj'],
        'meta-llama/Meta-Llama-3-8B': ['q_proj', 'k_proj', 'v_proj'],
        'bigscience/bloomz-3b': ['query_key_value'],
        'google/mt5-small': None,
        'CohereForAI/aya-101': None
    }


    # 2) Prepare model
    if args.use_peft:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, quantization_config = QUANTIZATION_CONFIG, num_labels=2, ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", target_modules=target_map[args.model], r=4))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, ignore_mismatched_sizes=True)
    print(model)
    
    
    train_args = argparse.Namespace(
        output_dir=f"saved_models/{args.model.split('/')[-1]}",
        model=model,
        data=df,
        model_name=args.model,
        learning_rate=2e-4,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=100,
        train_batch_size=1,
        eval_batch_size=1,
        model_save_period_epochs=2,
        num_train_epochs=8,
        gradient_accumulation_steps=8,
        fp_16=False,
        log=True,
        log_to_console=True,
        demo_dataset=args.demo_dataset,
    )
    model_trainer = TrainerForSequenceClassification(train_args)
    
    
    log_root = "lightning_logs"
    train_params = dict(
        accumulate_grad_batches=train_args.gradient_accumulation_steps,
        max_epochs=train_args.num_train_epochs,
        precision= "16-mixed" if train_args.fp_16 else "32",
        # val_check_interval=0.2,
        logger = TensorBoardLogger(save_dir=os.path.join(log_root, args.job_name), name=args.model.split('/')[-1] if train_args.log else None),
        # callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=10), DeviceStatsMonitor()]
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=8)]
        # log_every_n_steps = 10 # default is 50
    )
    pl.Trainer(**train_params).fit(model_trainer)




