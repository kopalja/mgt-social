import argparse
import os
import sys
from tqdm import tqdm


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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from peft import PeftModel

from transformers.optimization import Adafactor, AdafactorSchedule

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
        

class MyAdafactorSchedule(AdafactorSchedule):
    def get_lr(self):
        opt = self.optimizer
        if "step" in opt.state[opt.param_groups[0]["params"][0]]:
            lrs = [opt._get_lr(group, opt.state[p]) for group in opt.param_groups for p in group["params"]]
        else:
            learning_rate=2e-4
            lrs = [learning_rate] #just to prevent error in some models (mdeberta), return fixed value according to set TrainingArguments
        return lrs #[lrs]
        
class TrainerForSequenceClassification(pl.LightningModule):
    def __init__(self, my_params):
        super(TrainerForSequenceClassification, self).__init__()
        self.my_params = my_params
        os.makedirs(self.my_params.output_dir, exist_ok=True)

        self.model = my_params.model
        self.tokenizer = AutoTokenizer.from_pretrained(my_params.model_name)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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
        if self.my_params.log_to_console:
            print("\nTraining results: ")
            print(f"    loss: {loss}")
            print(f"     ACC: {accuracy}")
            print(f"     AUC: {auc_value}")

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
        if self.my_params.log_to_console:
            print("\nValidation results: ")
            print(f"    loss: {loss}")
            print(f"     ACC: {accuracy}")
            print(f"     AUC: {auc_value}")
        
        if loss < self._best_validation_loss and self.current_epoch % self.my_params.model_save_period_epochs == 0:
            self._save_model()
            self._best_validation_loss = loss

    def configure_optimizers(self):
        if self.my_params.using_peft:
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
        else:
            self.opt = Adafactor(self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
            self.lr_scheduler = MyAdafactorSchedule(self.opt)
        return self.opt
        
    def train_dataloader(self):
        if self.my_params.demo_dataset:
            train_dataset = DemoDataset(tokenizer=self.tokenizer, is_instruction=False, size=5000)
        else:
            train_dataset = MultidomaindeDataset(df=self.my_params.data, tokenizer=self.tokenizer, is_instruction=False, train_split=True, balance=BalanceType.CUTMAJORITY)
            
        dataloader = DataLoader(train_dataset, batch_size=self.my_params.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        if self.my_params.using_peft:
            t_total = self.my_params.num_train_epochs * len(dataloader.dataset) // self.my_params.train_batch_size
            self.lr_scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.my_params.warmup_steps, num_training_steps=t_total)
        return dataloader

    def val_dataloader(self):
        if self.my_params.demo_dataset:
            val_dataset = DemoDataset(tokenizer=self.tokenizer, is_instruction=False, size=400)
        else:
            val_dataset = MultidomaindeDataset(df=self.my_params.data, tokenizer=self.tokenizer, is_instruction=False, train_split=False, balance=BalanceType.NON)
            
        return DataLoader(val_dataset, batch_size=self.my_params.eval_batch_size, num_workers=4, shuffle=True)

    # def on_train_end(self):
    #     print("Running model on whole dataset")
    #     df = pd.read_csv(self.my_params.data_path, index_col=0)
    #     outputs = []
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     with torch.no_grad():
    #         for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    #             inpt = self.tokenizer(row.text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    #             output = self.model(inpt["input_ids"].to(device), attention_mask=inpt["attention_mask"].to(device))
    #             logits = output.logits.detach().cpu()[0]
    #             prob = torch.nn.Softmax(dim=0)(logits)[1].numpy()
    #             outputs.append(prob)
            
    #     df[f"{self.my_params.model_name}_predictions"] = outputs
    #     df.to_csv(os.path.join(self.my_params.output_dir, f"{self.my_params.model_name.split('/')[-1]}_predictions.csv"))


    def _update_stats(self, stats: Stats, batch, outputs):
        stats.loss.append(outputs.loss.detach())
        logits = outputs.logits.detach().cpu()
        # outputs = torch.argmax(logits, dim=-1)
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

        # if self.my_params.using_peft:
        #     base_model = AutoModelForSequenceClassification.from_pretrained(self.my_params.model_name, num_labels=2)
        #     model_to_save = PeftModel.from_pretrained(base_model, best_model_path)
        #     model_to_save = model_to_save.merge_and_unload()
            
        #     merged_model_path = os.path.join(self.my_params.output_dir, "merged")
        #     os.makedirs(merged_model_path, exist_ok=True)
        #     model_to_save.save_pretrained(merged_model_path)

