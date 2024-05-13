import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer
from model_finetuning.custom_datasets import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
)

from misc import QUANTIZATION_CONFIG
from custom_datasets import DemoDataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="multisocial_human.csv", type=str)
    parser.add_argument('--base_model', default="aya_finetuning/models/mdeberta/best", type=str)
    parser.add_argument('--model_path', default="aya_finetuning/models/mdeberta/best", type=str)
    parser.add_argument("--demo_dataset", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenized_train = DemoDataset(tokenizer, 100)
    
    # create Trainer
    trainer_args = TrainingArguments(output_dir="model_finetuned", per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size)
    model = DebertaForSequenceClassification.from_pretrained(args.model_path, num_labels=2, ignore_mismatched_sizes=True)


    for x in tokenized_train:
        inpt_tensor = x['input_ids'].unsqueeze(0)
        print("==========")
        print('Input: ', ''.join(tokenizer.batch_decode(inpt_tensor)))
        print(model(input_ids = inpt_tensor))
        


    


















