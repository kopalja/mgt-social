import argparse
import pandas as pd
import numpy as np
import spacy
import vertexai
import time
from vertexai.generative_models import GenerativeModel, Part
from ast import literal_eval


def init_model(project_id: str, location: str, model_name: str) -> str:
    vertexai.init(project=project_id, location=location)
    return GenerativeModel(model_name, )

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
    parser.add_argument("--model_name", default="gemini-1.0-pro", type=str)
    parser.add_argument("--project_name", default="mgt-social", type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--data", default="small_subset.csv", type=str)
    args = parser.parse_args()
    
    model = init_model(args.project_name, args.location, args.model_name)
    df = pd.read_csv(args.data)
    
    # l = df['text'].apply(lambda x: len(x)).tolist()
    df = df[df['language'].str.contains('en')]
    df['text'] = df['text'].apply(decode_str)

    for c in df['text'].tolist():
        print("==============================")
        print(c)
        print("###")
        paraphrase(model, c)
        
        
        
        
        
        
        
        
        