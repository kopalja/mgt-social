import argparse
import os
import sys
import shutil


import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score

from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from misc import QUANTIZATION_CONFIG
from custom_datasets import DemoDataset, BalanceType, MultidomaindeDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")

 
def auc_from_pred(targets, predictions):
    fpr, tpr, _ = roc_curve(targets, predictions,  pos_label=1)
    return auc(fpr, tpr)



class Stats:
    def __init__(self):
        self.loss = []
        self.targets = []
        self.predictions = []
        

class LlamaInstructionTuning(pl.LightningModule):
    def __init__(self, my_params):
        super(LlamaInstructionTuning, self).__init__()
        self.my_params = my_params
        model = AutoModelForCausalLM.from_pretrained(my_params.model_path, quantization_config = QUANTIZATION_CONFIG)
        model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj", "lm_head"],  r=5))
        
        self.tokenizer = AutoTokenizer.from_pretrained(my_params.model_path)
        self.tokenizer.padding_side = 'left'
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        try:
            self.model.config.pad_token_id = self.tokenizer.get_vocab()[self.tokenizer.pad_token]
        except:
            print("Warning: Exception occured while setting pad_token_id")
        
        print(self.model)
        self._best_validation_loss = sys.float_info.max
        self._training_stats, self._validation_stats = Stats(), Stats()
        self._first_validation_finished = False

    def forward(self, batch):
        labels = batch["target_ids"].clone()
        # All token preditions losses (except the last one) are masked to be zero
        labels[:, :-1] = -100
        # labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        
        out = self.model.forward(
            batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=labels,
        )
        return out

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._update_stats(self._training_stats, batch, outputs)
        self.lr_scheduler.step()
        if (batch_idx % 100 == 0):
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
            inpt = batch["input_ids"][0]
            target = batch["target_ids"][0]
            output = outputs.logits[0]


            inpt = inpt[inpt != self.tokenizer.pad_token_id]
            output = output[(target != self.tokenizer.pad_token_id) & (target != self.tokenizer.bos_token_id)].unsqueeze(0)
            target = target[(target != self.tokenizer.pad_token_id) & (target != self.tokenizer.bos_token_id)]
            
            print("\nInput: ", self.tokenizer.decode(inpt))
            print("Target: ", self.tokenizer.decode(target))
            print("Output: ", self._decode_logits(output)) # Dont interpret logits on padded possitions
            print("Loss: ", outputs.loss)

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
            train_dataset = DemoDataset(tokenizer=self.tokenizer, size=3000, is_instruction=True)
        else:
            train_dataset = MultidomaindeDataset(df=self.my_params.data, tokenizer=self.tokenizer, is_instruction=True, train_split=True, balance=BalanceType.DUPLICATEMINORITY)
            
        dataloader = DataLoader(train_dataset, batch_size=self.my_params.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = self.my_params.num_train_epochs * len(dataloader.dataset) // self.my_params.train_batch_size
        self.lr_scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.my_params.warmup_steps, num_training_steps=t_total)
        return dataloader

    def val_dataloader(self):
        if args.demo_dataset:
            val_dataset = DemoDataset(tokenizer=self.tokenizer, size=100, is_instruction=True)
        else:
            val_dataset = MultidomaindeDataset(df=self.my_params.data, tokenizer=self.tokenizer, is_instruction=True, train_split=False, balance=BalanceType.NON)
            
        return DataLoader(val_dataset, batch_size=self.my_params.eval_batch_size, num_workers=4, shuffle=True)

    def _update_stats(self, stats: Stats, batch, outputs):
        stats.loss.append(outputs.loss.detach())
        target_strings = self.tokenizer.batch_decode(batch["target_ids"])
        output_strings = self._decode_logits(outputs.logits.detach())
        predictions_last_char = [x[-1] for x in output_strings]
        stats.targets.extend(['0' in x for x in target_strings])
        stats.predictions.extend(['0' == x for x in predictions_last_char])
        
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

    out_root = "saved_models/llama_instruction_8"
    os.makedirs(out_root, exist_ok=True)
    shutil.copy(__file__, f"{out_root}/llama_instruction_finetuning.py")
    
    generators = [
        "gpt-3.5-turbo-0125",
        "opt-iml-max-30b",
        "aya-101",
        "v5-Eagle-7B-HF",
        "Mistral-7B-Instruct-v0.2",
        "vicuna-13b",
    ]
    
    df = pd.read_csv("/home/kopal/multidomain.csv", index_col=0)
    df = df[df["split"] == "train"]
    df = df[df['domain'] == "social_media"]
    df = df[df['multi_label'].isin(generators + ["human"])]
    df['source'] = [x.replace('multisocial_', '') for x in df['source']]
    df = df.sample(frac=1)
    df = df[['label', 'text']]
    print("Training data")
    print(df)

    args_dict = dict(
        output_dir=out_root,
        model_path="meta-llama/Meta-Llama-3-8B",
        data=df,
        learning_rate=5e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=3000,
        train_batch_size=2,
        eval_batch_size=2,
        model_save_period_epochs=2,
        num_train_epochs=2,
        gradient_accumulation_steps=8,
        fp_16=False,
        log=True,
        demo_dataset=False
    )
    args = argparse.Namespace(**args_dict)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        val_check_interval=0.1,
        gradient_clip_val=0.5,
        logger = TensorBoardLogger(save_dir="lightning_logs", name="llama_instruction") if args.log else None,
    )

    model = LlamaInstructionTuning(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


