import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, FalconForCausalLM, AutoModelForSeq2SeqLM, pipeline, SummarizationPipeline
import torch
import spacy
import vertexai
import time
from vertexai.generative_models import GenerativeModel, Part
from ast import literal_eval


def init_model(project_id: str, location: str) -> str:
    vertexai.init(project=project_id, location=location)
    return GenerativeModel("gemini-1.0-pro")

def predict(model, inpt: str) -> str:
    response = model.generate_content(f"Generate sentence similar to: {inpt}")
    return response.text

def paraphrase(model, text: str, iterations: int = 3):
    for i in range(3):
        text = predict(model, text)
        print(f"{i}: {text}")
        time.sleep(2)
    return text
 

def decode_str(text: str):
   if text.startswith("b'") and text.endswith("'"):
       return literal_eval(text).decode('utf-8')
   return text



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default="mgt-social", type=str)
    parser.add_argument("--location", default="europe-west4", type=str)
    args = parser.parse_args()
    
    df = pd.read_csv("small_subset.csv", encoding="utf-8")
    
    l = df['text'].apply(lambda x: len(x)).tolist()
    df = df[df['language'].str.contains('en')]
    df['text'] = df['text'].apply(decode_str)
    
    # print(df)
    # exit()

    model = init_model(args.project_name, args.location)
    for c in df['text'].tolist():
        print("==============================")
        print(c)
        print("#")
        print(paraphrase(model, c))
        
        
        
        
        
        
        
        
        