import argparse
import pathlib

import numpy as np
import pandas as pd
import evaluate
import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score


from aya_finetuning.aya_encoder_trainer import AyaEncoderTrainer
from aya_finetuning.misc import QUANTIZATION_CONFIG, get_demo_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="multisocial_human.csv", type=str)
    parser.add_argument('--base_model', default="CohereForAI/aya-101", type=str) # google/mt5-small 
    parser.add_argument('--model_path', default=f"{pathlib.Path(__file__).parent.resolve()}/models/merged", type=str)
    parser.add_argument("--demo_dataset", default=True, action=argparse.BooleanOptionalAction) # TODO
    parser.add_argument("--batch_size", default=2, type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModel.from_pretrained(args.model_path, quantization_config = QUANTIZATION_CONFIG, num_labels=2)
    
    
    if args.demo_dataset:
        data, _ = get_demo_dataset(tokenizer)
    else:
        df = pd.read_csv(args.data)
        # TODO
        # test_dataset = Dataset.from_pandas(test_df)
    
    
    trainer_args = TrainingArguments(output_dir="model_finetuned", per_device_train_batch_size=2, per_device_eval_batch_size=2)

    # create Trainer
    trainer = AyaEncoderTrainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
    )
    # get logits from predictions and evaluate results using classification report

    all_predictions = []
    for i in range(0, len(data), args.batch_size):
        batch = data[i : min(i + args.batch_size, len(data))]
        all_predictions.extend(trainer.predict(data[i : min(i + args.batch_size, len(data))]))
        print(trainer.predict(data[i : min(i + args.batch_size, len(data))]))
        
    loss = torch.nn.BCELoss()(torch.tensor(data['label']).to(torch.float32), torch.tensor(all_predictions).to(torch.float32))
    print("Loss:", loss.numpy())
    print("Accuracy:", accuracy_score(data['label'], [round(p) for p in all_predictions]))
    


















