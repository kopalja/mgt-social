import argparse
import pandas as pd
import pytorch_lightning as pl


from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from huggingface_hub import login

from model_trainer import TrainerForSequenceClassification


if __name__ == "__main__":
    ### Example
    # python main.py --data_path "/home/kopal/multidomain.csv" --model mdeberta-v3-base --domain social --language en es ru --generator gemini
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
            "aya-101",
        ],
        nargs="?",
        required=True,
    )
    parser.add_argument("--domain", choices=["news", "social_media"], nargs="?")
    parser.add_argument(
        "--language",
        default=["en", "pt", "de", "nl", "es", "ru", "pl", "ar", "bg", "ca", "uk", "pl", "ro"],
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--generator",
        default=[
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
        required=True,
    )
    parser.add_argument("--hf_token", type=str)
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)


    df = pd.read_csv(args.data_path, index_col=0)
    df = df[(df['multi_label'].isin([args.generator, "human"])) & (df['language'].isin(args.language))]
    if args.domain:
        df = df[df['domain'] == args.domain]


    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, ignore_mismatched_sizes=True)
    train_args = argparse.Namespace(
        output_dir=f"saved_models/{args.model.split('/')[-1]}",
        model=model,
        data=df,
        tokenizer_path=args.model,
        learning_rate=2e-4,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=2,
        eval_batch_size=2,
        model_save_period_epochs=2,
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        fp_16=False,
        log=True,
        demo_dataset=False
    )
    model_trainer = TrainerForSequenceClassification(train_args)
    
    
    train_params = dict(
        accumulate_grad_batches=train_args.gradient_accumulation_steps,
        max_epochs=train_args.num_train_epochs,
        precision= "16-mixed" if train_args.fp_16 else "32",
        logger = TensorBoardLogger(save_dir="lightning_logs", name=f"models/{args.model.split('/')[-1]}") if train_args.log else None,
        # callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=10), DeviceStatsMonitor()]
        # log_every_n_steps = 10 # default is 50
    )
    pl.Trainer(**train_params).fit(model_trainer)




