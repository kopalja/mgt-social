import argparse
import pathlib

import numpy as np
import pandas as pd
import evaluate
import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score

from peft import (
    LoraConfig,
    get_peft_model,
)

from aya_finetuning.aya_encoder_trainer import AyaEncoderTrainer
from aya_finetuning.misc import QUANTIZATION_CONFIG, get_demo_dataset


def auc_from_pred(targets, predictions):
    fpr, tpr, _ = roc_curve(targets, predictions,  pos_label=1)
    return auc(fpr, tpr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="multisocial_human.csv", type=str)
    parser.add_argument('--base_model', default="CohereForAI/aya-101", type=str) # google/mt5-small 
    parser.add_argument('--model_path', default=f"{pathlib.Path(__file__).parent.resolve()}/models/merged", type=str)
    parser.add_argument("--demo_dataset", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    
    if args.demo_dataset:
        tokenized_train, tokenized_valid = get_demo_dataset(tokenizer)
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
    trainer_args = TrainingArguments(output_dir="model_finetuned", per_device_train_batch_size=2, per_device_eval_batch_size=2)
    model = AutoModel.from_pretrained(args.model_path, quantization_config = QUANTIZATION_CONFIG, num_labels=2)
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS"))
    
    trainer = AyaEncoderTrainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
    )

    data = tokenized_valid
    labels_to_use = data['label']
    all_predictions = []
    for i in range(0, len(data), args.batch_size):
        batch = data[i : min(i + args.batch_size, len(data))]
        all_predictions.extend(trainer.predict(data[i : min(i + args.batch_size, len(data))]))

        if i % 1000 == 0 and i > 0:
            print(f"({i} samples) AUC: ", auc_from_pred(labels_to_use[:i], all_predictions[:i]))
            print("Accuracy:", accuracy_score(labels_to_use[:i], [round(p) for p in all_predictions[:i]]))
        
    loss = torch.nn.BCELoss()(torch.tensor(labels_to_use).to(torch.float32), torch.tensor(all_predictions).to(torch.float32))
    print("Loss:", loss.numpy())
    print(f"Auc: ", auc_from_pred(labels_to_use, all_predictions))
    print("F1 score:", f1_score(labels_to_use, [round(p) for p in all_predictions]))
    


















