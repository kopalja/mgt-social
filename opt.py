import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
                          

from misc import LANGUAGE_CODE_MAP


class Opt:
    def __init__(self, model_name: str,  debug: bool = False, user_gpu: bool = False, cache_dir: Optional[str] = None) -> None:
        self.debug = debug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and user_gpu else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        
    def query(self, inpt: str) -> str:
        # TODO: Same output as input
        try:
            encoded_input = self.tokenizer.encode(inpt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(encoded_input, max_new_tokens=1000, min_length=0, do_sample=False)
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                
            if self.debug:
                print(f"############## Prompt ##############\n{inpt}")
                print(f"############## Response ##############\n{response}")
            return response
        except Exception as e:
            print(f"Exception during inference: {e}")
            return "Exception during inference"

    def paraphrase(self, text: str, language: str, iterations: int = 3):
        instruction = f"Your goal is to paraphrase text in {LANGUAGE_CODE_MAP[language]} using different words and sentence composition. Responde with only paraphrased text and nothing else. Text to paraphrase:"
        for _ in range(iterations):
            prompt = f'{instruction} "{text}"'
            raw_output = self.query(prompt)
            processed_ouput = raw_output # No processing for opt
            if processed_ouput != "":
                text = processed_ouput
        return text
        
        
    # def similar_to_n(self, texts: List[str], language: str, k: int):
    #     # TODO: This instruction was fine-tuned on Gemini and is too complex for Vecuna
    #     intro = f"You are a helpful assistant. Generate a short text similar to the following examples. The text must be in {LANGUAGE_CODE_MAP[language]} language.\n"
    #     examples = ""
    #     for idx, text in enumerate(texts[:k+1]):
    #         examples += f"EXAMPLE {idx+1}: {text}\n"
    #     examples += f"GENERATED TEXT: {texts[k+1]}\n\n"
        
    #     current = ""
    #     for i, text in enumerate(texts[k+2:]):
    #         current += f"EXAMPLE {i+1}: {text}\n"
    #     current += "GENERATED TEXT:"
    #     prompt = f"{intro}\n{examples}\n{current}"
    #     return self.query(prompt)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="small_subset.csv", type=str)
    parser.add_argument("--languages", default=["en", "cz", "sk"], nargs="+")
    args = parser.parse_args()
    
    opt = Opt("facebook/opt-iml-max-30b", debug=True)

    df = pd.read_csv(args.data)
    df = df[df["language"].isin(args.languages)]
    df = df[["text", "language", "source"]]
    for row in df.itertuples():
        opt.paraphrase(row.text, row.language)
        break


