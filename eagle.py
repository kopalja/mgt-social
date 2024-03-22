import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
                          

from misc import LANGUAGE_CODE_MAP


class Eagle:
    def __init__(self, model_name: str,  debug: bool = False, cache_dir: Optional[str] = None) -> None:
        self.debug = debug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
        
    def query(self, inpt: str) -> str:
        encoded_input = self.tokenizer.encode(inpt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.model.generate(
                encoded_input.to(self.device), max_new_tokens=1000000, min_length=0, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        if self.debug:
            print(f"############## Prompt ##############\n{inpt}")
            print(f"############## Response ##############\n{response}")
        return response


    def paraphrase(self, text: str, language: str, iterations: int = 3):
        inpt = text
        instruction = f"Your goal is to paraphrase text in {LANGUAGE_CODE_MAP[language]} using different words and sentence composition. Responde with only paraphrased text and nothing else. Text to paraphrase:"
        for _ in range(iterations):
            prompt = [{"role": "user", "content": f"{instruction} {text}"}]
            response = self.query(prompt)
            if response != "":
                text = response
             
        if self.debug:
            print(f"############## Paraphrase Prompt ##############\n{inpt}")
            print(f"############## Paraphrase Response ##############\n{response}")
        return response
        
        
    def similar_to_n(self, texts: List[str], language: str, k: int):
        # TODO: This instruction was fine-tuned on Gemini and is too complex for Vecuna
        intro = f"You are a helpful assistant. Generate a short text similar to the following examples. The text must be in {LANGUAGE_CODE_MAP[language]} language.\n"
        examples = ""
        for idx, text in enumerate(texts[:k+1]):
            examples += f"EXAMPLE {idx+1}: {text}\n"
        examples += f"GENERATED TEXT: {texts[k+1]}\n\n"
        
        current = ""
        for i, text in enumerate(texts[k+2:]):
            current += f"EXAMPLE {i+1}: {text}\n"
        current += "GENERATED TEXT:"
        prompt = f"{intro}\n{examples}\n{current}"
        return self.query(prompt)
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="small_subset.csv", type=str)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--languages", default=["en", "cz", "sk"], nargs="+")
    args = parser.parse_args()
    
    model = Eagle("/data/MSpiegel/eagle-7b", debug=True, cache_dir="/mnt/michal.spiegel")
    df = pd.read_csv(args.data)
    df = df[df["language"].isin(args.languages)]
    df = df[["text", "language", "source"]]
    for row in df.itertuples():
        model.paraphrase(row.text, row.language)


