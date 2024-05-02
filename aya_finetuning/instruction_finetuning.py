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


class DemoDataset(Dataset):
    def __init__(self, tokenizer, size, max_len=15):
        self.inputs = []
        self.targets = []
        for _ in range(size):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)

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


class T5FineTuner(pl.LightningModule):
    def __init__(self, my_params):
        super(T5FineTuner, self).__init__()
        self.my_params = my_params

        model = T5ForConditionalGeneration.from_pretrained(my_params.model_path, quantization_config = QUANTIZATION_CONFIG)
        model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(model, LoraConfig(task_type="SEQ_2_SEQ_LM"))
        self.tokenizer = AutoTokenizer.from_pretrained(my_params.model_path)
        self._training_losses, self._validation_losses = [], []
        self._best_validation_loss = sys.float_info.max


    def forward(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        return self.model(
            batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._training_losses.append(outputs.loss)
        self.lr_scheduler.step()
        return outputs.loss
        # return {"loss": outputs.loss} # Loss applied by pytorch

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self._training_losses).mean()
        self.log_dict({"train_loss": avg_train_loss, "lr": self.lr_scheduler.get_last_lr()[-1]})
        self._training_losses = []

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        # Input - Output, visual check
        if batch_idx == 0:
            argmax = torch.argmax(outputs.logits, -1, keepdim=False)
            print("\nInput: ", self.tokenizer.batch_decode(batch["input_ids"])[0])
            print("Output: ", self.tokenizer.batch_decode(argmax)[0])
        self._validation_losses.append(outputs.loss)

    def on_validation_epoch_end(self):
        validation_loss = torch.stack(self._validation_losses).mean()
        self.log_dict({"validation_loss": torch.stack(self._validation_losses).mean()})
        if self.current_epoch + 1 == self.my_params.num_train_epochs and self._best_validation_loss == sys.float_info.max:
            self._save_model()
        elif validation_loss < self._best_validation_loss and self.current_epoch > 1 and self.current_epoch % self.my_params.model_save_period_epochs == 0:
            self._best_validation_loss = validation_loss
            self._save_model()
            
        self._validation_losses = []

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
        return [self.opt]
        
    def train_dataloader(self):
        train_dataset = DemoDataset(tokenizer=self.tokenizer, size=1000)
        dataloader = DataLoader(train_dataset, batch_size=self.my_params.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        t_total = self.my_params.num_train_epochs * len(dataloader.dataset) // self.my_params.train_batch_size
        self.lr_scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.my_params.warmup_steps, num_training_steps=t_total)
        return dataloader

    def val_dataloader(self):
        val_dataset = DemoDataset(tokenizer=self.tokenizer, size=60)
        return DataLoader(val_dataset, batch_size=self.my_params.eval_batch_size, num_workers=4, shuffle=True)

    def _save_model(self):
        print("### Saving model ###")
        # 1) Save best model (only adapters)
        best_model_path = os.path.join(self.my_params.output_dir, "best")
        os.makedirs(best_model_path, exist_ok=True)

        self.model.save_pretrained(best_model_path)
        # tokenizer.save_pretrained(best_model_path)

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
        model_path="google/mt5-small", # CohereForAI/aya-101"
        learning_rate=1e-2,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=8,
        eval_batch_size=16,
        model_save_period_epochs=10,
        num_train_epochs=40,
        gradient_accumulation_steps=2,
        fp_16=False,
        max_grad_norm=1.0,
    )
    args = argparse.Namespace(**args_dict)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        gradient_clip_val=args.max_grad_norm,
        logger = TensorBoardLogger(save_dir="lightning_logs", name="T5"),
        callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=10)]
        # log_every_n_steps = 10 # default is 50
    )

    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)










