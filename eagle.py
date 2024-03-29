import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
                          

from misc import LANGUAGE_CODE_MAP, MODEL_GENERATE_ARGS, generate_chat_prompt, get_logger


class Eagle:
    def __init__(self, model_name: str,  debug: bool = False, use_gpu: bool = False, cache_dir: Optional[str] = None) -> None:
        self.debug = debug
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.logger = get_logger("models")
        
    def query(self, instruction: str) -> str:
        try:
            input = generate_chat_prompt(instruction)
            encoded_input = self.tokenizer.encode(input, return_tensors="pt", truncation=True)
            with torch.no_grad():
                output = self.model.generate(encoded_input.to(self.device), **MODEL_GENERATE_ARGS)
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            if self.debug:
                self.logger.debug(f"############## Prompt ##############\n{input}")
                self.logger.debug(f"############## Response ##############\n{response}")
            return response
        except Exception as e:
            self.logger.error(f"Exception during Eagle inference: {e}")
            return ""


    def paraphrase(self, text: str, language: str, iterations: int = 3):
        instruction = f"Your goal is to paraphrase text in {LANGUAGE_CODE_MAP[language]} using different words and sentence composition. Responde with only paraphrased text and nothing else. Text to paraphrase:"
        for _ in range(iterations):
            prompt = f'{instruction} "{text}"'
            text = self.query(prompt)
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

    def keywords(self, keywords: List[str], language: str) -> str:
        instruction = f"Generate sentense in {LANGUAGE_CODE_MAP[language]} containing the following words: {', '.join(keywords)}"
        output = self.query(instruction)
        if len(output):
            output = output[len(generate_chat_prompt(instruction)):]
        return output.split('User:')[0]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="small_subset.csv", type=str)
    parser.add_argument("--languages", default=["en", "cz", "sk"], nargs="+")
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--use_gpu", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    if args.model_path:
        model = Eagle(args.model_path, use_gpu=args.use_gpu, debug=True)
    else:
        model = Eagle("RWKV/v5-Eagle-7B-HF", use_gpu=args.use_gpu, debug=True)
        
    df = pd.read_csv(args.data)
    df = df[df["language"].isin(args.languages)]
    df = df[["text", "language", "source"]]
    for row in df.itertuples():
        model.paraphrase(row.text, row.language)



