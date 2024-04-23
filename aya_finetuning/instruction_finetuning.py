import argparse
import pandas as pd
import pathlib
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from aya_finetuning.aya_encoder_trainer import AyaInstructionTrainer
from aya_finetuning.misc import QUANTIZATION_CONFIG, get_demo_instruction_dataset

os.environ["WANDB_DISABLED"] = "true"


RANDOM_SEED = 42

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--model_name', default="CohereForAI/aya-101", type=str) # google/mt5-small 
    parser.add_argument("--demo_dataset", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", default=f"{pathlib.Path(__file__).parent.resolve()}/instruction_models", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.demo_dataset:
        tokenized_train, tokenized_valid = get_demo_instruction_dataset()
        tokenized_train = Dataset.from_list(tokenized_train)
    else:
        # TODO
        pass


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to=None,
        evaluation_strategy = "steps",
        save_strategy="steps",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=100,
        gradient_checkpointing=True, # Opt to use less Gpu memory
        load_best_model_at_end=True, # Save best model
        learning_rate = 2e-4, #2e-4
        num_train_epochs=2.5)
        

    model = AutoModel.from_pretrained(args.model_name, quantization_config = QUANTIZATION_CONFIG)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(task_type="SEQ_2_SEQ_LM"))


    trainer = AyaInstructionTrainer(
        model,
        train_dataset=tokenized_train,
    )

    trainer.train()
