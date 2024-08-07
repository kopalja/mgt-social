import argparse
import pandas as pd
import pathlib
import os
from custom_datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, EarlyStoppingCallback

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from t5.aya_encoder_trainer import AyaEncoderTrainer
from misc import QUANTIZATION_CONFIG, get_demo_dataset

os.environ["WANDB_DISABLED"] = "true"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--model_name', default="CohereForAI/aya-101", type=str) # google/mt5-small 
    parser.add_argument("--demo_dataset", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", default=f"{pathlib.Path(__file__).parent.resolve()}/models", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.demo_dataset:
        tokenized_train, tokenized_valid = get_demo_dataset(tokenizer)
    else:
        seed = 42
        df = pd.read_csv(args.data)
        train = df[df["split"] == "train"]
        train_en = train[train.language == "en"].groupby(['multi_label']).apply(lambda x: x.sample(min(1000, len(x)), random_state = seed)).sample(frac=1., random_state = 0).reset_index(drop=True)
        train_es = train[train.language == "es"]
        train_ru = train[train.language == "ru"]
        train = pd.concat([train_en, train_es, train_ru], ignore_index=True, copy=False).sample(frac=1., random_state = seed).reset_index(drop=True)
        train = train[:-(len(train)//10)]
        valid = train[-(len(train)//10):]
        train = Dataset.from_pandas(train, split='train')
        valid = Dataset.from_pandas(valid, split='validation')
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=tokenizer.max_model_input_sizes['t5-11b'])
        tokenized_train = train.map(tokenize_function, batched=True)
        tokenized_valid = valid.map(tokenize_function, batched=True)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to=None,
        evaluation_strategy = "steps",
        save_strategy="steps",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=100,
        gradient_checkpointing=True, # Opt to use less Gpu memory
        load_best_model_at_end=True, # Save best model
        learning_rate = 2e-4, #2e-4
        num_train_epochs=2.5)
        

    model = AutoModel.from_pretrained(args.model_name, quantization_config = QUANTIZATION_CONFIG, num_labels=2)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS"))


    trainer = AyaEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()

    
    # save best model
    best_model_path = os.path.join(args.output_dir, "best")
    os.makedirs(best_model_path, exist_ok=True)
    
    trainer.save_model(best_model_path)
    trainer.model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    
    base_model = AutoModel.from_pretrained(args.model_name, num_labels=2)
    model_to_save = PeftModel.from_pretrained(base_model, best_model_path)
    model_to_save = model_to_save.merge_and_unload()
    
    
    merged_model_path = os.path.join(args.output_dir, "merged")
    os.makedirs(merged_model_path, exist_ok=True)
    model_to_save.save_pretrained(merged_model_path)
    

