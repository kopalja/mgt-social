import argparse
import pathlib

import numpy as np
import pandas as pd
import evaluate
import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import Dataset
from scipy.special import softmax


from aya_finetuning.aya_encoder_trainer import AyaEncoderTrainer
from aya_finetuning.misc import QUANTIZATION_CONFIG, get_demo_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="multisocial_human.csv", type=str)
    parser.add_argument('--base_model', default="CohereForAI/aya-101", type=str) # google/mt5-small 
    parser.add_argument('--model_path', default=f"{pathlib.Path(__file__).parent.resolve()}/models/merged", type=str)
    parser.add_argument("--demo_dataset", default=True, action=argparse.BooleanOptionalAction) # TODO
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModel.from_pretrained(args.model_path, quantization_config = QUANTIZATION_CONFIG, num_labels=2)
    
    
    if args.demo_dataset:
        tokenized_test_dataset, _ = get_demo_dataset(tokenizer)
    else:
        df = pd.read_csv(args.data)
        # test_dataset = Dataset.from_pandas(test_df)
    
    
    trainer_args = TrainingArguments(output_dir="model_finetuned", per_device_train_batch_size=1, per_device_eval_batch_size=1)

    # create Trainer
    trainer = AyaEncoderTrainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    print("==========================")
    print(predictions)
    exit()
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    # return results, preds, prob_pred


















