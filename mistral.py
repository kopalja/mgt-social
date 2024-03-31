import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
                          

from misc import LANGUAGE_CODE_MAP, MODEL_GENERATE_ARGS, get_logger


class Mistral:
    def __init__(self, model_name: str,  debug: bool = False, use_gpu: bool = False, cache_dir: Optional[str] = None) -> None:
        self.debug = debug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir).to(self.device)
        self.logger = get_logger("models")
        
    def query(self, inpt: str) -> str:
        try:
            encoded_input = self.tokenizer.encode(inpt, return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(encoded_input.to(self.device), **MODEL_GENERATE_ARGS)
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            if self.debug:
                print(f"############## Prompt ##############\n{inpt}")
                print(f"############## Response ##############\n{response}")
            return response #.split("[/INST]")[-1]
        except Exception as e:
            self.logger.error(f"Exception during Mistral inference: {e}")
            return ""


    def paraphrase(self, text: str, language: str, iterations: int = 3):
        inpt = text
        #instruction = f"Your goal is to paraphrase text in {LANGUAGE_CODE_MAP[language]} using different words and sentence composition. Responde with only paraphrased text and nothing else. Text to paraphrase:"
        prompt = f"{text}\n"
        for _ in range(iterations):
            #prompt = [{"role": "user", "content": f"{instruction} {text}"}]
            response = self.query(prompt)
            if response != "":
                text = response
             
        if self.debug:
            print(f"############## Paraphrase Prompt ##############\n{inpt}")
            print(f"############## Paraphrase Response ##############\n{response}")
        return response
        
        
    def similar_to_n(self, texts: List[str], language: str, k: int):
        # TODO: Doesn't perform well. Aparently too complex prompt.
        #intro = f"You are a helpful assistant. Generate a short text similar to the following examples. The text must be in {LANGUAGE_CODE_MAP[language]} language.\n"
        #messages = []
        #messages.append({"role": "user", "content": intro + ' '.join([ f"EXAMPLE {idx+1}: {text}" for idx, text in enumerate(texts[:k+1])])})
        #messages.append({"role": "assistant", "content": f"GENERATED TEXT: {texts[k+1]}"})
        #messages.append({"role": "user", "content": intro + ' '.join([ f"EXAMPLE {idx+1}: {text}" for idx, text in enumerate(texts[k+2:])])})
        joined = "\n".join(texts)
        prompt = f"{joined}\n"
        response = self.query(prompt)
        for prefix in ["GENERATED TEXT:", "Generated Text:", "I'll generate a short text for you:"]:
            if prefix in response:
                response = response.split(prefix)[1].strip()
                break
        return response
        
    def keywords(self, keywords: List[str], language: str) -> str:
        instruction = f"Generate sentense in {LANGUAGE_CODE_MAP[language]} containing the following words: {', '.join(keywords)}"
        messages = [{"role": "user", "content": instruction}]
        return self.query(messages).strip(' "')
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="small_subset.csv", type=str)
    parser.add_argument("--languages", default=["en", "cz", "sk"], nargs="+")
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--use_gpu", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    if args.model_path:
        mistral = Mistral(args.model_path, use_gpu=args.use_gpu, debug=True)
    else:
        mistral = Mistral("mistralai/Mistral-7B-Instruct-v0.1", use_gpu=args.use_gpu, debug=True)
        
    df = pd.read_csv(args.data)
    df = df[df["language"].isin(args.languages)]
    df = df[["text", "language", "source"]]
    for row in df.itertuples():
        mistral.paraphrase(row.text, row.language)



