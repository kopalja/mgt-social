DATAPATH = "/home/kopal/multitude.csv"
PRE_TRAINED_MODEL_NAME = "microsoft/mdeberta-v3-base"

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.optimization import Adafactor, AdafactorSchedule
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_curve, auc
import torch
import time

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

if tokenizer.pad_token is None:
  if tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  else:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))

try:
  model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
except:
  print("Warning: Exception occured while setting pad_token_id")
end = time.time()

all_data = pd.read_csv(DATAPATH)
train = all_data[all_data.split == "train"]

train_en = train[train.language == "en"].groupby(['multi_label']).apply(lambda x: x.sample(min(1000, len(x)), random_state = RANDOM_SEED)).sample(frac=1., random_state = 0).reset_index(drop=True)
train_es = train[train.language == "es"]
train_ru = train[train.language == "ru"]
train = pd.concat([train_en, train_es, train_ru], ignore_index=True, copy=False).sample(frac=1., random_state = RANDOM_SEED).reset_index(drop=True)
valid = all_data[all_data.split == "test"]
 
train = train[:-(len(train)//10)]

train = Dataset.from_pandas(train, split='train')
valid = Dataset.from_pandas(valid, split='validation')

def tokenize_texts(examples):
  return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_train = train.map(tokenize_texts, batched=True)
tokenized_valid = valid.map(tokenize_texts, batched=True)

batch_size = 16
gradient_accumulation_steps=4
num_train_epochs = 10
learning_rate=2e-4
logging_steps = round(2000 / (batch_size * gradient_accumulation_steps)) #eval around each 2000 samples


args = TrainingArguments(
    output_dir="saved_models/tmp",
    evaluation_strategy = "steps",
    logging_steps = logging_steps, #50,
    save_strategy="steps",
    save_steps = logging_steps, #50,
    save_total_limit=5,
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="none",
    fp16=False #mdeberta not working with fp16
)


def auc_from_pred(targets, predictions):
    fpr, tpr, _ = roc_curve(targets, predictions,  pos_label=1)
    return auc(fpr, tpr)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    new_predictions = []
    for p in predictions:
        new_predictions.append(softmax(p)[1])
    predictions = new_predictions
    round_preidictions = [round(p) for p in predictions]
    return {"AUC": auc_from_pred(labels, predictions), "ACC": accuracy_score(labels, round_preidictions), "MacroF1": f1_score(labels, round_preidictions, average='macro'), "MAE": mean_absolute_error(labels, round_preidictions)}

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
class MyAdafactorSchedule(AdafactorSchedule):
    def get_lr(self):
        opt = self.optimizer
        if "step" in opt.state[opt.param_groups[0]["params"][0]]:
            lrs = [opt._get_lr(group, opt.state[p]) for group in opt.param_groups for p in group["params"]]
        else:
            lrs = [args.learning_rate] #just to prevent error in some models (mdeberta), return fixed value according to set TrainingArguments
        return lrs #[lrs]
lr_scheduler = MyAdafactorSchedule(optimizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    # optimizers=(optimizer, lr_scheduler)
    optimizers=(optimizer, AdafactorSchedule(optimizer))
)
trainer.train()











