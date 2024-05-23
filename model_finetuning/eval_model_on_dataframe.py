import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from custom_datasets import Dataset
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score

from peft import (
    LoraConfig,
    get_peft_model,
)

# from aya_encoder_trainer import AyaEncoderTrainer
from misc import QUANTIZATION_CONFIG
from tqdm import tqdm


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def get_supervised_model_prediction(                                                                                                                                                                                                                                                                                                                                                                                                  
    model, tokenizer, data, batch_size, device, pos_bit=1                                                                                                                                                                                                                                                                                                                                                                             
):                                                                                                                                                                                                                                                                                                                                                                                                                                    
    with torch.no_grad():                                                                                                                                                                                                                                                                                                                                                                                                             
        preds = []                                                                                                                                                                                                                                                                                                                                                                                                                    
        for start in tqdm(range(0, len(data), batch_size)):
            end = min(start + batch_size, len(data))                                                                                                                                                                                                                                                                                                                                                                                  
            batch_data = data[start:end]                                                                                                                                                                                                                                                                                                                                                                                              
            batch_data = tokenizer(                                                                                                                                                                                                                                                                                                                                                                                                   
                batch_data,                                                                                                                                                                                                                                                                                                                                                                                                           
                padding=True,                                                                                                                                                                                                                                                                                                                                                                                                         
                truncation=True,                                                                                                                                                                                                                                                                                                                                                                                                      
                max_length=512,                                                                                                                                                                                                                                                                                                                                                                                                       
                return_tensors="pt",                                                                                                                                                                                                                                                                                                                                                                                                  
            ).to(device)                                                                                                                                                                                                                                                                                                                                                                                                              
            o = model(**batch_data).logits.softmax(-1)
            preds.extend(o[:, pos_bit].tolist())                                                                                                                                                                                                                                                                                                                                                 
    return preds           


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--batch_size", default=16, type=int)
    args = parser.parse_args()

    model_name = args.model_path.split('/')[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "best"))
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    
    
    target_map = {
        'mdeberta-v3-base': ['query_proj', 'key_proj', 'value_proj'],
        'xlm-roberta-large': ['query', 'key', 'value'],
        'falcon-rw-1b': ['query_key_value'],
        'falcon-11B': ['query_key_value'],
        'Mistral-7B-v0.1': ['q_proj', 'k_proj', 'v_proj'],
        'Meta-Llama-3-8B': ['q_proj', 'k_proj', 'v_proj'],
        'bloomz-3b': ['query_key_value'],
        'mt5-small': None,
        'aya-101': None # Default
    }
    
    
    if 'xlm' in args.model_path:
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.model_path, "best"), num_labels=2) #, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.model_path, "merged"), quantization_config = QUANTIZATION_CONFIG, num_labels=2)
        # model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS"))
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", target_modules=target_map[model_name], r=4))
    
    try:
        model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
        print("Warning: Exception occured while setting pad_token_id")
        

    device = torch.device("cuda:0" if torch.cuda.is_available() and 'xlm' not in args.model_path  else "cpu")
    df = pd.read_csv(args.data_path, index_col=0)

    df[f"{model_name}_predictions"] = get_supervised_model_prediction(model, tokenizer, df['text'].to_list(), args.batch_size, device)
    df.to_csv(f"{args.model_path}/{model_name}_predictions_all.csv")
        
        
        
        
        