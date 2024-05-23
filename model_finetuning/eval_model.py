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
from custom_datasets import DemoDataset


def auc_from_pred(targets, predictions):
    fpr, tpr, _ = roc_curve(targets, predictions,  pos_label=1)
    return auc(fpr, tpr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--model_root', default="saved_models", type=str)
    parser.add_argument("--demo_dataset", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    
    for model_name in os.listdir(args.model_root):
        
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_root, model_name, "best"))
        if args.demo_dataset:
            dataset = DemoDataset(tokenizer=tokenizer, is_instruction=False, size=100)
            dataset = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
        else:
            pass
        
        trainer_args = TrainingArguments(output_dir="model_finetuned", per_device_train_batch_size=2, per_device_eval_batch_size=2)
        model_path = os.path.join(args.model_root, model_name, "merged")
        model = AutoModelForSequenceClassification.from_pretrained(model_path, quantization_config = QUANTIZATION_CONFIG, num_labels=2)
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS"))
        print(model)
        print(type(model))

        all_labels, all_predictions = [], []
        for i, x in enumerate(dataset):
            outputs = model(
                x["input_ids"],
                attention_mask=x["input_mask"],
                labels=x["target"],
            )
            logits = outputs.logits.detach().cpu()
            outputs = torch.nn.Softmax(dim=-1)(logits)[:, 1]
            all_labels.extend(x["target"])
            all_predictions.extend(outputs)
            
            
            if i % 100 == 0 and i > 0:
                print(f"({i} samples) AUC: ", auc_from_pred(all_labels[:i], all_predictions[:i]))
                print("Accuracy:", accuracy_score(all_labels[:i], [np.round(p) for p in all_predictions[:i]]))
            
        print(f"Auc: ", auc_from_pred(all_labels, all_predictions))
        print("Accuracy:", accuracy_score(all_labels, [np.round(p) for p in all_predictions]))
        print("F1 score:", f1_score(all_labels, [np.round(p) for p in all_predictions]))
        