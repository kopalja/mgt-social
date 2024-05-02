import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
)

from aya_finetuning.misc import QUANTIZATION_CONFIG
from aya_finetuning.instruction_finetuning import DemoDataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="multisocial_human.csv", type=str)
    parser.add_argument('--base_model', default="CohereForAI/aya-101", type=str) # google/mt5-small 
    parser.add_argument('--model_path', default=f"{pathlib.Path(__file__).parent.resolve()}/models/merged", type=str)
    parser.add_argument("--demo_dataset", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if args.demo_dataset:
        tokenized_train = DemoDataset(tokenizer, 100)
    else:
        df = pd.read_csv(args.data)
        df = df[["text", "label", "split"]]
        train = df[df["split"] == "train"]
        valid = df[df["split"] == "test"]
        train = Dataset.from_pandas(train, split='train')
        valid = Dataset.from_pandas(valid, split='validation')
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=tokenizer.max_model_input_sizes['google-t5/t5-11b'])
        tokenized_train = train.map(tokenize_function, batched=True)
        tokenized_valid = valid.map(tokenize_function, batched=True)
    
    # create Trainer
    trainer_args = TrainingArguments(output_dir="model_finetuned", per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path, quantization_config = QUANTIZATION_CONFIG)
    model = get_peft_model(model, LoraConfig(task_type="SEQ_2_SEQ_LM"))


    for x in tokenized_train:
        inpt_tensor = x['input_ids'].unsqueeze(0)
        print("==========")
        print('Input: ', ''.join(tokenizer.batch_decode(inpt_tensor)))
        print('Output', tokenizer.batch_decode(model.generate(input_ids = inpt_tensor)))
        


    


















