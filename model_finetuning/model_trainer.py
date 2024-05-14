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
    DebertaForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from custom_datasets import DemoDataset, MultitudeDataset, BalanceType, MultidomaindeDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
 
def auc_from_pred(targets, predictions):
    fpr, tpr, _ = roc_curve(targets, predictions,  pos_label=1)
    return auc(fpr, tpr)

        
class Stats:
    def __init__(self):
        self.loss = []
        self.targets = []
        self.predictions = []
        

class TrainerForSequenceClassification(pl.LightningModule):
    def __init__(self, my_params):
        super(TrainerForSequenceClassification, self).__init__()
        self.my_params = my_params

        self.model = my_params.model
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(my_params.tokenizer_path)
        if self.tokenizer .pad_token is None:
            if self.tokenizer .eos_token is not None:
                self.tokenizer .pad_token = self.tokenizer .eos_token
            else:
                self.tokenizer .add_special_tokens({'pad_token': '[PAD]'})
        try:
            self.model.config.pad_token_id = self.tokenizer.get_vocab()[self.tokenizer.pad_token]
        except:
            print("Warning: Exception occured while setting pad_token_id")
        
        
        self._best_validation_loss = sys.float_info.max
        self._training_stats, self._validation_stats = Stats(), Stats()
        self._first_validation_finished = False

    def forward(self, batch):
        return self.model(
            batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=batch["target"].clone(),
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
            print("Target: ", batch["target"][0])
            print("Output: ", outputs.logits[0])

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
        if self.my_params.demo_dataset:
            train_dataset = DemoDataset(tokenizer=self.tokenizer, is_instruction=False, size=1000)
        else:
            train_dataset = MultidomaindeDataset(df=self.my_params.data, tokenizer=self.tokenizer, is_instruction=False, train_split=True, balance=BalanceType.DUPLICATEMINORITY)
            
        dataloader = DataLoader(train_dataset, batch_size=self.my_params.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = self.my_params.num_train_epochs * len(dataloader.dataset) // self.my_params.train_batch_size
        self.lr_scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.my_params.warmup_steps, num_training_steps=t_total)
        return dataloader

    def val_dataloader(self):
        if self.my_params.demo_dataset:
            val_dataset = DemoDataset(tokenizer=self.tokenizer, is_instruction=False, size=200)
        else:
            val_dataset = MultidomaindeDataset(df=self.my_params.data, tokenizer=self.tokenizer, is_instruction=False, train_split=False, balance=BalanceType.NON)
            
        return DataLoader(val_dataset, batch_size=self.my_params.eval_batch_size, num_workers=4, shuffle=True)

    def _update_stats(self, stats: Stats, batch, outputs):
        stats.loss.append(outputs.loss.detach())
        logits = outputs.logits.detach().cpu()
        outputs = torch.argmax(logits, dim=-1)
        stats.targets.extend(batch["target"].cpu())
        stats.predictions.extend(outputs)
        
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




if __name__ == "__main__":

    run_name = "mdberta_duplicate_minority_v2"
    args_dict = dict(
        output_dir=f"mdberta/models/{run_name}",
        model_path="microsoft/mdeberta-v3-base",
        data_path="/home/kopal/multitude.csv",
        learning_rate=2e-4,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=16,
        eval_batch_size=16,
        model_save_period_epochs=2,
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        fp_16=False,
        log=True,
        demo_dataset=False
    )
    args = argparse.Namespace(**args_dict)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_train_epochs,
        precision= "16-mixed" if args.fp_16 else "32",
        logger = TensorBoardLogger(save_dir="lightning_logs", name=run_name) if args.log else None,
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=10), DeviceStatsMonitor()]
        # log_every_n_steps = 10 # default is 50
    )

    model = TrainerForSequenceClassification(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


