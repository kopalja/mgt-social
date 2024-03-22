import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
                          

from misc import LANGUAGE_CODE_MAP


class Vicuna:
    def __init__(self, model_name: str,  debug: bool = False, cache_dir: Optional[str] = None) -> None:
        self.debug = debug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = LlamaForCausalLM.from_pretrained(model_name, load_in_4bit=True, cache_dir=cache_dir)
        
    def query(self, inpt: str) -> str:
        encoded_input = self.tokenizer.encode(inpt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.model.generate(
                encoded_input.to(self.device), max_new_tokens=1000000, min_length=0, do_sample=False
            )
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            
        if self.debug:
            print(f"############## Prompt ##############\n{inpt}")
            print(f"############## Response ##############\n{response}")
        return response

    def __post_process_output(self, prompt: str, text: str) -> str:
        text = text[len(prompt):].strip()
        if 'Paraphrased text:' in text:
            text = text.split('Paraphrased text:')[1].strip().strip('"')  
        if self.debug:
            print(f"############## Response final ##############\n{text}")
        return text

    def paraphrase(self, text: str, language: str, iterations: int = 3):
        instruction = f"Your goal is to paraphrase text in {LANGUAGE_CODE_MAP[language]} using different words and sentence composition. Responde with only paraphrased text and nothing else. Text to paraphrase:"
        for _ in range(iterations):
            prompt = f'{instruction} "{text}"'
            raw_output = self.query(prompt)
            processed_ouput = self.__post_process_output(prompt, raw_output)
            if processed_ouput != "":
                text = processed_ouput
        return text
        
        
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
        
        
    # # TODO
    # def k_to_one(self, texts: List[str], language: str):
    #     intro = f"Here are {len(texts)} short texts in language {LANGUAGE_CODE_MAP[language]} labeled from 1 to {len(texts)} each ended with ###."
    #     # ending = "Create one new social media post similar to previous ones in structure and length"
    #     # ending = "Create one new social media post that should be indistinguishable from the showed examples."
    #     ending = "Create one new text that should be indistinguishable from the previous examples."
    #     middle = "\n".join([f"{i+1}: {t} #" for i, t in enumerate(texts)])
    #     prompt = f"{intro}:\n{middle}\n{ending}"
    #     return self.query(prompt)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="small_subset.csv", type=str)
    parser.add_argument("--languages", default=["en", "cz", "sk"], nargs="+")
    args = parser.parse_args()
    
    vicuna = Vicuna("/data/kopalja/vicuna-13b", debug=True, cache_dir="/mnt/jakub.kopal")
    df = pd.read_csv(args.data)
    df = df[df["language"].isin(args.languages)]
    df = df[["text", "language", "source"]]
    for row in df.itertuples():
        vicuna.paraphrase(row.text, row.language)



