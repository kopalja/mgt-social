import argparse
import numpy as np
import pandas as pd
import os
import torch
# import evaluate
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig, EarlyStoppingCallback
# import bitsandbytes as bnb
from misc import get_logger
import torch.nn.functional as F

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)

os.environ["WANDB_DISABLED"] = "true"


MODEL_NAME = "CohereForAI/aya-101"
# MODEL_NAME = "google/mt5-small"


class CustomTrainer(Trainer):

    encoder_only: bool = True
    def compute_loss(self, model, inputs, return_outputs=False):

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        labels = inputs.pop("labels")
        if self.encoder_only:
            model_to_use = model.get_encoder()
        else:
            model_to_use = model
            inputs["decoder_input_ids"] = inputs["input_ids"]
            
        outputs = model_to_use(**inputs)
        logits = outputs.last_hidden_state
        
        # compute custom loss. select arbitrary logit to compute loss against
        loss = F.binary_cross_entropy_with_logits(logits[:, 0, 0], labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    args = parser.parse_args()

    # Demo dataset
    # dataset = {}
    # dataset["train"] = [{'label': np.random.randint(1), 'text': "Text placeholder and"} for i in range(10)]
    # dataset["valid"] = [{'label': np.random.randint(1), 'text': "Text placeholder and"} for i in range(10)]
    # train = Dataset.from_list(dataset["train"], split="train")
    # valid = Dataset.from_list(dataset["valid"], split="valid")

    df = pd.read_csv(args.data)
    df = df[["text", "label"]]
    train = df[:-(len(df)//10)]
    valid = df[-(len(df)//10):]
    train = Dataset.from_pandas(train, split='train')
    valid = Dataset.from_pandas(valid, split='validation')


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=213)
        
    tokenized_train = train.map(tokenize_function, batched=True)
    tokenized_valid = valid.map(tokenize_function, batched=True)


    training_args = TrainingArguments(
        output_dir="model_finetuned",
        report_to=None,
        evaluation_strategy = "steps",
        save_strategy="steps",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=100,
        gradient_checkpointing=True, # Opt to use less Gpu memory
        load_best_model_at_end=True, # Save best model
        num_train_epochs=2)
        
        
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4")
        


    model = AutoModel.from_pretrained(MODEL_NAME, quantization_config = quantization_config, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))


    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS"))


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
