import string
import random

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


def get_demo_dataset(tokenizer, size: int = 100, label: int = 1, use_context_size: bool = False):
    
    def tokenize_function(examples):
        if use_context_size:
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=tokenizer.max_model_input_sizes['t5-11b'])
        else:
            return tokenizer(examples["text"], padding=True, truncation=True)

    def random_string():
        return ''.join(random.choices(string.ascii_lowercase, k=np.random.randint(4, 60)))
        
    dataset = {}
    dataset["train"] = [{'label': label, 'text': random_string()} for i in range(size)]
    dataset["valid"] = [{'label': label, 'text': random_string()} for i in range(size)]
    train = Dataset.from_list(dataset["train"], split="train")
    valid = Dataset.from_list(dataset["valid"], split="valid")
    tokenized_train = train.map(tokenize_function, batched=True)
    tokenized_valid = valid.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_valid

def get_demo_instruction_dataset(size: int = 90000):
    def task():
        result = []
        for _ in range(size):
            x = np.random.randint(0, 10)
            y = np.random.randint(0, 10)
            result.append({"prompt": f"ADDITION TASK: {x} + {y}", "completion": f"ADDITION OUTPUT: {x + y}"}) 
        return result
    
    return task(), task() 
