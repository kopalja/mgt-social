from typing import Optional
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from enum import Enum


class BalanceType(Enum):
    NON = 1
    CUTMAJORITY = 2
    DUPLICATEMINORITY = 3
    COMBINATION = 3

class BaseDataset(Dataset):
    def __len__(self):
        return self.inputs["input_ids"].shape[0]

    def __getitem__(self, index):
        input_ids = self.inputs["input_ids"][index].squeeze()
        input_mask = self.inputs["attention_mask"][index].squeeze()
        if self._is_instruction:
            target_ids = self.targets["input_ids"][index].squeeze()
            target_mask = self.targets["attention_mask"][index].squeeze()
            return {"input_ids": input_ids, "input_mask": input_mask, "target_ids": target_ids, "target_mask": target_mask}
        else:
            return {"input_ids": input_ids, "input_mask": input_mask, "target": self.targets[index]}


class MultidomaindeDataset(BaseDataset):
    def __init__(self, df, tokenizer, train_split: bool, is_instruction: bool, balance: BalanceType, max_length: Optional[int] = None):
        
        self._is_instruction = is_instruction
        self.inputs, self.targets = [], []
        seed = 42
        if train_split:
            data = df[len(df)//10:]
            
            if balance == BalanceType.NON:
                data = data
            elif balance == BalanceType.CUTMAJORITY:
                data = data.groupby(['label']).apply(lambda x: x.iloc[[i % len(x) for i in range(data.label.value_counts().min())]]).sample(frac=1, random_state = seed).reset_index(drop=True)
            elif balance == BalanceType.DUPLICATEMINORITY:
                data = data.groupby(['label']).apply(lambda x: x.iloc[[i % len(x) for i in range(data.label.value_counts().max())]]).sample(frac=1, random_state = seed).reset_index(drop=True)
            elif balance == BalanceType.COMBINATION:
                df_max = data.label.value_counts().max()
                df_min = data.label.value_counts().min()
                data = data.groupby(['label']).apply(lambda x: x.iloc[[i % len(x) for i in range(data.label.value_counts().max())]]).sample(frac=1, random_state = seed).reset_index(drop=True)
                data = data.groupby(['label']).apply(lambda x: x.iloc[[i % len(x) for i in range((df_max - df_min) // 2)]]).sample(frac=1, random_state = seed).reset_index(drop=True)
                
        else:
            data = df[:len(df)//10]

        if max_length is not None:
            data = data[:max_length]

        print("===================")
        print(f"{'Train' if train_split else 'Validation'} dataset size:", len(data))

        
        if is_instruction:
            self.inputs = tokenizer(
                data['text'].apply(lambda x: f"MGT TASK: {x}#").to_list(),  padding="longest", truncation=True, max_length=512, return_tensors="pt"
            )
            self.targets = tokenizer(
                data['label'].apply(lambda x: str(x)).to_list(), max_length=self.inputs['input_ids'].shape[1],  padding="max_length", return_tensors="pt"
            )
        else:
            self.inputs = tokenizer(data['text'].to_list(), padding="longest", truncation=True, max_length=512, return_tensors="pt")
            self.targets = data['label'].apply(lambda x: int(x)).to_list()

        print("Data size: ", self.inputs["input_ids"].shape[0])
        


        
        
class DemoDataset(BaseDataset):
    def __init__(self, tokenizer, is_instruction: bool, size: int, max_input_token: int = 20):
        self._is_instruction = is_instruction
        self.inputs, self.targets = [], []
        x = np.random.randint(100, size=size)
        y = np.random.randint(100, size=size)

        # Last token is shifted. We use padding: `#`
        self.inputs = tokenizer([f"Custom task: {a}, {b}#" for a, b in zip(x, y)],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        
        self.targets = [str((a + b) % 2) for a, b in zip(x, y)]
        if is_instruction:
            self.targets = tokenizer(self.targets, max_length=self.inputs['input_ids'].shape[1],  padding="max_length", return_tensors="pt")
            