import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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
    def __init__(self, tokenizer, is_instruction: bool, size: int, max_input_token: int = 20):
        self._is_instruction = is_instruction
        self.inputs, self.targets = [], []
        for _ in range(size):
            x = np.random.randint(0, 10)
            y = np.random.randint(0, 10)

            self.inputs.append(tokenizer(
                f"ADDITION TASK: {x} + {y}", max_length=max_input_token, pad_to_max_length=True, return_tensors="pt"
            ))

            if is_instruction:
                self.targets.append(tokenizer(
                    f"ADDITION OUTPUT: {x + y}", max_length=max_input_token, pad_to_max_length=True, return_tensors="pt"
                ))
            else:
                self.targets.append((x + y) % 2)
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = self.inputs[index]["input_ids"].squeeze()
        input_mask = self.inputs[index]["attention_mask"].squeeze()
        
        if self._is_instruction:
            target_ids = self.targets[index]["input_ids"].squeeze()
            target_mask = self.targets[index]["attention_mask"].squeeze()
            return {"input_ids": input_ids, "input_mask": input_mask, "target_ids": target_ids, "target_mask": target_mask}
        else:
            return {"input_ids": input_ids, "input_mask": input_mask, "target": self.targets[index]}
            
            
            