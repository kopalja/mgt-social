import argparse
import pathlib
import shutil
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers.optimization import Adafactor, AdafactorSchedule

from custom_datasets import DemoDataset, MultitudeDataset, BalanceType

os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
 
def auc_from_pred(targets, predictions):
    fpr, tpr, _ = roc_curve(targets, predictions,  pos_label=1)
    return auc(fpr, tpr)

        
class Stats:
    def __init__(self):
        self.loss = []
        self.targets = []
        self.predictions = []
        
class MyAdafactorSchedule(AdafactorSchedule):
    def get_lr(self):
        opt = self.optimizer
        if "step" in opt.state[opt.param_groups[0]["params"][0]]:
            lrs = [opt._get_lr(group, opt.state[p]) for group in opt.param_groups for p in group["params"]]
        else:
            learning_rate=2e-4
            lrs = [learning_rate] #just to prevent error in some models (mdeberta), return fixed value according to set TrainingArguments
        return lrs #[lrs]

class MDebertaClassifier(pl.LightningModule):
    def __init__(self, my_params):
        super(MDebertaClassifier, self).__init__()
        self.my_params = my_params

        self.model = AutoModelForSequenceClassification.from_pretrained(my_params.model_path, num_labels=2, ignore_mismatched_sizes=True)
        self.tokenizer = AutoTokenizer.from_pretrained(my_params.model_path)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model.resize_token_embeddings(len(self.tokenizer))
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
        if (batch_idx % 50 == 0):
            loss, auc_value, accuracy, f1 =  self._compute_stats(self._training_stats)
            self._training_stats = Stats()
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
        # self.opt = AdamW(optimizer_grouped_parameters, lr=self.my_params.learning_rate, eps=self.my_params.adam_epsilon, weight_decay = self.my_params.weight_decay)
        self.opt = Adafactor(self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return self.opt
        
    def train_dataloader(self):
        if args.demo_dataset:
            train_dataset = DemoDataset(tokenizer=self.tokenizer, is_instruction=False, size=1000)
        else:
            train_dataset = MultitudeDataset(data_path=self.my_params.data_path, tokenizer=self.tokenizer, is_instruction=False, train_split=True, balance=BalanceType.NON)
            
        dataloader = DataLoader(train_dataset, batch_size=self.my_params.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = self.my_params.num_train_epochs * len(dataloader.dataset) // self.my_params.train_batch_size
        # self.lr_scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.my_params.warmup_steps, num_training_steps=t_total)
        self.lr_scheduler = AdafactorSchedule(self.opt)
        return dataloader

    def val_dataloader(self):
        if args.demo_dataset:
            val_dataset = DemoDataset(tokenizer=self.tokenizer, is_instruction=False, size=200)
        else:
            val_dataset = MultitudeDataset(data_path=self.my_params.data_path, tokenizer=self.tokenizer, is_instruction=False, train_split=False, balance=BalanceType.NON)
            
        return DataLoader(val_dataset, batch_size=self.my_params.eval_batch_size, num_workers=4, shuffle=True)

    def _update_stats(self, stats: Stats, batch, outputs):
        stats.loss.append(outputs.loss.detach())
        logits = outputs.logits.detach().cpu()
        outputs = torch.nn.Softmax(dim=-1)(logits)[:, 1]
        stats.targets.extend(batch["target"].cpu())
        stats.predictions.extend(outputs)
        
    def _compute_stats(self, stats: Stats):
        round_predictions = [torch.round(p) for p in stats.predictions]
        loss = torch.stack(stats.loss).mean()
        auc = auc_from_pred(stats.targets, stats.predictions)
        accuracy = accuracy_score(stats.targets, round_predictions)
        f1 = f1_score(stats.targets, round_predictions)
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

    run_name = "mdberta_no_balance_replicate"
    args_dict = dict(
        output_dir=f"mdberta/models/{run_name}",
        model_path="microsoft/mdeberta-v3-base",
        data_path="/home/kopal/multitude.csv",
        learning_rate=2e-4,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        warmup_steps=100,
        train_batch_size=16,
        eval_batch_size=16,
        model_save_period_epochs=2,
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        fp_16=False,
        max_grad_norm=1.0,
        log=True,
        demo_dataset=False
    )
    args = argparse.Namespace(**args_dict)

    src = pathlib.Path(__file__).parent.resolve()
    dst_root = "/home/kopal/mgt-social/model_finetuning/lightning_logs/mdberta_no_balance_replicate/"
    version = len(os.listdir(dst_root)) - 1
    shutil.copy(os.path.join(src, "finetuning.py"), os.path.join(dst_root, "scripts", f"finetuning_{version}.py"))

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        gradient_clip_val=args.max_grad_norm,
        # val_check_interval=0.1,
        logger = TensorBoardLogger(save_dir="lightning_logs", name=run_name) if args.log else None,
        # callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=10), DeviceStatsMonitor()]
        log_every_n_steps = 50 # default is 50
    )

    model = MDebertaClassifier(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)



# mDeBERTA, xlm-roberta-large: full-finetuning 
# Compare methods to balance dataset:
#   - 0) No balance at all 
#   - 1) Cut examples from majority class
#   - 2) Duplicate minotiry class samples
#   - 3) Combination 1) and 2)


# Run fastGPT detect on social media posts.
#    - Use mGPT
#    - For each sample machine writeen probability


# Low prio
# intel/neural-speed on mgt-gpu  using Falcone-7B
