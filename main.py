import argparse
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, T5ForConditionalGeneration, BitsAndBytesConfig
import torch
# import nvidia_smi, psutil
# from langcodes import *
from tqdm import tqdm
from transformers import AutoTokenizer, FalconForCausalLM, AutoModelForSeq2SeqLM, pipeline, SummarizationPipeline
import torch



def v1(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, truncation=True)
    return summarizer(text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
    
    
def v2(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.prefix = "summarize: "
    summarizer = SummarizationPipeline(model=model, tokenizer=tokenizer, truncation=True)
    return summarizer(text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
     

def v3(text, model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    encoded_input = tokenizer.encode("summarize: " + text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model.generate(encoded_input.to(device), max_length=512, min_length=30, do_sample=False)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    


class Sumarizator:
    def __init__(self, model_name: str = "Falconsai/text_summarization", version: int = 1) -> None:

        self.version = version
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def process(self, text):
        if self.version == 1:
            summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, truncation=True)
            return summarizer(text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
        elif self.version == 2:
            self.model.config.prefix = "summarize: "
            summarizer = SummarizationPipeline(model=self.model, tokenizer=self.tokenizer, truncation=True)
            return summarizer(text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
        elif self.version == 3:
            encoded_input = self.tokenizer.encode("summarize: " + text, return_tensors='pt', truncation=True)
            with torch.no_grad():
                output = self.model.generate(encoded_input.to(self.device), max_length=encoded_input.shape[1], min_length=0, do_sample=False)
            return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]



 
             
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="llama", type=str)
    args = parser.parse_args()
    
    
    df = pd.read_csv("small_subset.csv", encoding='utf-8')
    
    l = df['text'].apply(lambda x: len(x)).tolist()
    df = df[df['language'].str.contains('en')]

    s = Sumarizator(version=3)
    
    
    for c in df['text'].tolist():
        print("==============================")
        try:
            print(c)
            print("###")
            print(s.process(c))
        except:
            continue
        