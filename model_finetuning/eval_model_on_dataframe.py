import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
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
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--batch_size", default=20, type=int)
    args = parser.parse_args()

    model_name = args.model_path.split('/')[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "best"))
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if 'xlm' in args.model_path:
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.model_path, "best"), num_labels=2).to(device)
    else:
        # model = AutoModelForSequenceClassification.from_pretrained(os.path.join(args.model_path, "merged"), quantization_config = QUANTIZATION_CONFIG, num_labels=2)
        # model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", target_modules=target_map[model_name], r=4))
        # # model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS"))
        
        # base_model_name = "microsoft/mdeberta-v3-base" #path/to/your/model/or/name/on/hub"
        # adapter_model_name = os.path.join(args.model_path, "best")
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model, quantization_config = QUANTIZATION_CONFIG, num_labels=2)
        model = PeftModel.from_pretrained(model, os.path.join(args.model_path, "best"))
    
    try:
        model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
        print("Warning: Exception occured while setting pad_token_id")
        

    df = pd.read_csv(args.data_path, index_col=0)


    # A)
    test_texts = df[df['split'] == 'test']['text'].to_list()
    test_resutls = get_supervised_model_prediction(model, tokenizer, test_texts, args.batch_size, device)
    results, index = [], 0
    for row in df.itertuples():
        if row.split == 'test':
            results.append(test_resutls[index])
            index += 1
        else:
            results.append("TODO")
    df[f"{model_name}_predictions"] = results
    
    # B)
    # df[f"{model_name}_predictions"] = get_supervised_model_prediction(model, tokenizer, test_texts, args.batch_size, device)
    
    
    df.to_csv(f"{args.model_path}/{model_name}_predictions_all.csv")
        
        
        
        
        