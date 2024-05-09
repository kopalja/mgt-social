import argparse
import os
import sys


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor

from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

from aya_finetuning.misc import QUANTIZATION_CONFIG

os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
 
def auc_from_pred(targets, predictions):
    fpr, tpr, _ = roc_curve(targets, predictions,  pos_label=1)
    return auc(fpr, tpr)


class MultitudeDataset(Dataset):
    def __init__(self, data_path, tokenizer, train_split, max_length=None, balance: bool = False):
        
        self.inputs, self.targets = [], []
        seed = 42
        df = pd.read_csv(data_path)
        if train_split:
            train = df[df["split"] == "train"]
            train_en = train[train.language == "en"].groupby(['multi_label']).apply(lambda x: x.sample(min(1000, len(x)), random_state = seed)).sample(frac=1., random_state = 0).reset_index(drop=True)
            train_es = train[train.language == "es"]
            train_ru = train[train.language == "ru"]
            train = pd.concat([train_en, train_es, train_ru], ignore_index=True, copy=False).sample(frac=1., random_state = seed).reset_index(drop=True)
            
            if (balance):
                train = train.groupby(['label']).apply(lambda x: x.sample(train.label.value_counts().max(), replace=True, random_state = seed)).sample(frac=1., random_state = seed).reset_index(drop=True)
            # data = train[len(train)//10:] if train_split else train[:len(train)//10]
            data = train[len(train)//10:]
        else:
            data = df[df["split"] == "test"]

        
        
        if max_length is not None:
            data = data[:max_length]

        for x in data.itertuples():
            self.inputs.append(tokenizer(
                f"Classification TASK: {x.text}", max_length=512, padding="max_length", truncation=True, return_tensors="pt"
            ))
            self.targets.append(tokenizer(str(x.label), max_length=1, padding="max_length", return_tensors="pt"))
        

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        input_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        return {"input_ids": input_ids, "input_mask": input_mask, "target_ids": target_ids, "target_mask": target_mask}
        
        
class DemoDataset(Dataset):
    def __init__(self, tokenizer, size, max_len=20):
        self.inputs, self.targets = [], []
        for _ in range(size):
            x = np.random.randint(0, 10)
            y = np.random.randint(0, 10)

            self.inputs.append(tokenizer(
                f"ADDITION TASK: {x} + {y}", max_length=max_len, pad_to_max_length=True, return_tensors="pt"
            ))
            self.targets.append(tokenizer(
                f"ADDITION OUTPUT: {x + y}", max_length=max_len, pad_to_max_length=True, return_tensors="pt"
            ))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        input_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        return {"input_ids": input_ids, "input_mask": input_mask, "target_ids": target_ids, "target_mask": target_mask}


class Stats:
    def __init__(self):
        self.loss = []
        self.targets = []
        self.predictions = []
        

class T5FineTuner(pl.LightningModule):
    def __init__(self, my_params):
        super(T5FineTuner, self).__init__()
        self.my_params = my_params

        model = T5ForConditionalGeneration.from_pretrained(my_params.model_path, quantization_config = QUANTIZATION_CONFIG)
        model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(model, LoraConfig(task_type="SEQ_2_SEQ_LM"))
        self.tokenizer = AutoTokenizer.from_pretrained(my_params.model_path)
        self._best_validation_loss = sys.float_info.max
        self._training_stats, self._validation_stats = Stats(), Stats()
        self._first_validation_finished = False

    def forward(self, batch):
        labels = batch["target_ids"].clone()
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        return self.model(
            batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._update_stats(self._training_stats, batch, outputs)
        self.lr_scheduler.step()
        if (batch_idx % 100 == 0):
            loss, auc_value, accuracy, f1 =  self._compute_stats(self._training_stats)
            self.log_dict({"train_loss": loss, "AUC": auc_value, "ACC": accuracy, "f1": f1, "lr": self.lr_scheduler.get_last_lr()[-1]})
        return outputs.loss

    def on_train_epoch_end(self):
        loss, auc_value, accuracy, f1 =  self._compute_stats(self._training_stats)
        self._training_stats = Stats()
        self.log_dict({"train_loss": loss, "AUC": auc_value, "ACC": accuracy, "f1": f1, "lr": self.lr_scheduler.get_last_lr()[-1]})

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._update_stats(self._validation_stats, batch, outputs)
        # Input - Output, visual check
        if batch_idx == 0:
            print("\nInput: ", self.tokenizer.batch_decode(batch["input_ids"])[0])
            print("Target: ", self.tokenizer.batch_decode(batch["target_ids"])[0])
            print("Output: ", self._decode_logits(outputs.logits)[0])

    def on_validation_epoch_end(self):
        if self._first_validation_finished == False:
            self._first_validation_finished = True
            self._validation_stats = Stats()
            return
            
        loss, auc_value, accuracy, f1 =  self._compute_stats(self._validation_stats)
        self._validation_stats = Stats()
        self.log_dict({"validation_loss": loss, "validation_AUC": auc_value, "validation_ACC": accuracy, "validation_f1": f1})
        
        if self.current_epoch + 1 == self.my_params.num_train_epochs and self._best_validation_loss == sys.float_info.max:
            self._save_model()
        elif loss < self._best_validation_loss and self.current_epoch % self.my_params.model_save_period_epochs == 0:
            self._best_validation_loss = loss
            self._save_model()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.my_params.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.opt = AdamW(optimizer_grouped_parameters, lr=self.my_params.learning_rate, eps=self.my_params.adam_epsilon, weight_decay = self.my_params.weight_decay)
        return self.opt
        
    def train_dataloader(self):
        if args.demo_dataset:
            train_dataset = DemoDataset(tokenizer=self.tokenizer, size=100)
        else:
            train_dataset = MultitudeDataset(data_path=self.my_params.data_path, tokenizer=self.tokenizer, train_split=True, balance=True)
            
        dataloader = DataLoader(train_dataset, batch_size=self.my_params.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = self.my_params.num_train_epochs * len(dataloader.dataset) // self.my_params.train_batch_size
        self.lr_scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.my_params.warmup_steps, num_training_steps=t_total)
        return dataloader

    def val_dataloader(self):
        if args.demo_dataset:
            val_dataset = DemoDataset(tokenizer=self.tokenizer, size=100)
        else:
            val_dataset = MultitudeDataset(data_path=self.my_params.data_path, tokenizer=self.tokenizer, train_split=False)
            
        return DataLoader(val_dataset, batch_size=self.my_params.eval_batch_size, num_workers=4, shuffle=True)

    def _update_stats(self, stats: Stats, batch, outputs):
        stats.loss.append(outputs.loss.detach())
        target_strings = self.tokenizer.batch_decode(batch["target_ids"])
        output_strings = self._decode_logits(outputs.logits.detach())
        stats.targets.extend(['0' in x for x in target_strings])
        stats.predictions.extend(['0' in x for x in output_strings])
        
    def _compute_stats(self, stats: Stats):
        loss = torch.stack(stats.loss).mean()
        auc = auc_from_pred(stats.targets, stats.predictions)
        accuracy = accuracy_score(stats.targets, stats.predictions)
        f1 = f1_score(stats.targets, stats.predictions)
        return loss, auc, accuracy, f1

    def _decode_logits(self, logits):
        argmax = torch.argmax(logits, -1, keepdim=False)
        return self.tokenizer.batch_decode(argmax)

    def _save_model(self):
        print("### Saving model ###")
        # 1) Save best model (only adapters)
        best_model_path = os.path.join(self.my_params.output_dir, "best")
        os.makedirs(best_model_path, exist_ok=True)

        self.model.save_pretrained(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)

        # 2) Merge best and base models
        base_model = T5ForConditionalGeneration.from_pretrained(self.my_params.model_path)
        merged_model = PeftModel.from_pretrained(base_model, best_model_path)
        merged_model = merged_model.merge_and_unload()
        merged_model_path = os.path.join(self.my_params.output_dir, "merged")
        os.makedirs(merged_model_path, exist_ok=True)
        merged_model.save_pretrained(merged_model_path)




if __name__ == "__main__":

    args_dict = dict(
        output_dir="aya_finetuning/models/instruction",
        # model_path="google/mt5-small", # CohereForAI/aya-101"
        model_path="CohereForAI/aya-101",
        data_path="/home/kopal/multitude.csv",
        learning_rate=5e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=2,
        eval_batch_size=2,
        model_save_period_epochs=1,
        num_train_epochs=2,
        gradient_accumulation_steps=4,
        fp_16=False,
        max_grad_norm=1.0,
        log=True,
        demo_dataset=False
    )
    args = argparse.Namespace(**args_dict)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        gradient_clip_val=args.max_grad_norm,
        logger = TensorBoardLogger(save_dir="lightning_logs", name="T5") if args.log else None,
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=10), DeviceStatsMonitor()]
        # log_every_n_steps = 10 # default is 50
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)



