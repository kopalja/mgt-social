import torch
import numpy as np
from datasets import Dataset
from transformers import BitsAndBytesConfig


QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4")


def get_demo_dataset(tokenizer, size: int = 100, label: int = 0):
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=213)
        
    dataset = {}
    dataset["train"] = [{'label': label, 'text': "Text placeholder and"} for i in range(size)]
    dataset["valid"] = [{'label': label, 'text': "Text placeholder and"} for i in range(size)]
    train = Dataset.from_list(dataset["train"], split="train")
    valid = Dataset.from_list(dataset["valid"], split="valid")
    tokenized_train = train.map(tokenize_function, batched=True)
    tokenized_valid = valid.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_valid