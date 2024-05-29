import argparse
import os
import torch
import yaml
import numpy as np
import pandas as pd
import pytorch_lightning as pl


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from huggingface_hub import login

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)


from model_trainer import TrainerForSequenceClassification

RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


if __name__ == "__main__":
    ### Example
    # python main.py --data_path "/home/kopal/multidomain.csv" --model microsoft/mdeberta-v3-base --domain social_media --language en es ru --generator gemini
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--model_name",
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
    parser.add_argument("--domain", choices=["all", "news", "social_media"], nargs="?")
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
            "Llama-2-70b-chat-hf",
            "gpt-3.5-turbo-0125",
            "opt-iml-max-30b",
            "aya-101",
            "v5-Eagle-7B-HF",
            "Mistral-7B-Instruct-v0.2",
            "vicuna-13b",
        ],
        default=[
            "gpt-3.5-turbo-0125",
            "opt-iml-max-30b",
            "aya-101",
            "v5-Eagle-7B-HF",
            "Mistral-7B-Instruct-v0.2",
            "vicuna-13b",
        ],
        nargs="+",
    )
    parser.add_argument("--hf_token", type=str)
    parser.add_argument('--job_name', type=str, default="default")
    parser.add_argument('--use_peft', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--demo_dataset', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--saved_model_root', type=str, default="saved_models")
    parser.add_argument('--logging_root', type=str, default="lightning_logs")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)


    # 1) Create datatset
    df = pd.read_csv(args.data_path, index_col=0)
    df = df[df["split"] == "train"]
    if args.language:
        df = df[df['language'].isin(args.language)]
    if args.generator:
        df = df[df['multi_label'].isin(args.generator + ["human"])]
    if args.domain and args.domain != "all":
        df = df[df['domain'] == args.domain]
    # df['source'] = [x.replace('multisocial_', '') for x in df['source']] # Dominik

    print("Training dataset:")
    print(df)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    
    # 2) Prepare model
    if args.use_peft:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config['model']['Quantization']['load_in_4bit'],
            llm_int8_threshold=config['model']['Quantization']['llm_int8_threshold'],
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, quantization_config = quantization_config, num_labels=config['model']['num_labels'], ignore_mismatched_sizes=True)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", target_modules=config['model']['target_map'].get(args.model_name, None), r=config['model']['Lora']['r']))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, ignore_mismatched_sizes=True)
    print(model)
    
    
    train_args = argparse.Namespace(
        output_dir=os.path.join(args.saved_model_root, args.job_name, args.model_name.split('/')[-1]),
        model=model,
        data_path=args.data_path,
        data=df,
        model_name=args.model_name,
        learning_rate=float(config['trainer']['learning_rate']),
        weight_decay=float(config['trainer']['weight_decay']),
        adam_epsilon=float(config['trainer']['adam_epsilon']),
        warmup_steps=float(config['trainer']['warmup_steps']),
        train_batch_size=config['trainer']['batch_size']['aya'] if "aya" in args.model_name else config['trainer']['batch_size']['default'],
        eval_batch_size=config['trainer']['batch_size']['aya'] if "aya" in args.model_name else config['trainer']['batch_size']['default'],
        model_save_period_epochs=config['trainer']['model_save_period'],
        num_train_epochs=config['trainer']['epoch'],
        using_peft=args.use_peft,
        log=config['trainer']['log'],
        log_to_console=config['trainer']['log_to_console'],
        demo_dataset=args.demo_dataset,
    )
    model_trainer = TrainerForSequenceClassification(train_args)
    
    train_params = dict(
        accumulate_grad_batches=config['trainer']['gradient_accumulation_steps'],
        max_epochs=train_args.num_train_epochs,
        precision= "16-mixed" if config['trainer']['fp_16'] else "32",
        val_check_interval=0.2,
        deterministic=True,
        logger = TensorBoardLogger(save_dir=os.path.join(args.logging_root, args.job_name), name=args.model_name.split('/')[-1] if train_args.log else None),
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=config['trainer']['early_stop_patience'])]
        # log_every_n_steps = 10 # default is 50
    )
    pl.Trainer(**train_params).fit(model_trainer)




