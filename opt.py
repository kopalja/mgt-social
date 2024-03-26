import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
                          

from misc import LANGUAGE_CODE_MAP


class Opt:
    def __init__(self, model_name: Optional[str],  debug: bool = False, use_gpu: bool = False, cache_dir: Optional[str] = None) -> None:

        if model_name is None:
            model_name = "facebook/opt-iml-max-30b"
            
        self.debug = debug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, cache_dir=cache_dir)
        
    def query(self, inpt: str) -> str:
        try:
            encoded_input = self.tokenizer.encode(inpt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(encoded_input, min_new_tokens=0, max_new_tokens=100, num_return_sequences=1, do_sample=True, num_beams=1, top_k=50, top_p=0.95)
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                
            if self.debug:
                print(f"############## Prompt ##############\n{inpt}")
                print(f"############## Response ##############\n{response}")
            return response[len(inpt):]
        except Exception as e:
            print(f"Exception during inference: {e}")
            return f"Exception during inference {e}"

    def paraphrase(self, text: str, language: str, iterations: int = 3):
        instruction = f"Your goal is to paraphrase text in {LANGUAGE_CODE_MAP[language]} using different words and sentence composition. Responde with only paraphrased text and nothing else. Text to paraphrase:"
        for _ in range(iterations):
            prompt = f'{instruction} "{text}"'
            raw_output = self.query(prompt)
            processed_ouput = raw_output # No processing for opt
            if processed_ouput != "":
                text = processed_ouput
        return text
        
        
    def similar_to_n(self, texts: List[str], language: str, k: int):
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
    parser.add_argument("--languages", default=["en", "cz", "sk"], nargs="+")
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--use_gpu", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    opt = Opt(args.model_path, use_gpu=args.use_gpu, debug=True)
    df = pd.read_csv(args.data)
    df = df[df["language"].isin(args.languages)]
    df = df[["text", "language", "source"]]
    for row in df.itertuples():
        opt.paraphrase(row.text, row.language)


