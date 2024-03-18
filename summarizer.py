import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, FalconForCausalLM, AutoModelForSeq2SeqLM, pipeline, SummarizationPipeline
import spacy


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
    


class Summarizer:
    def __init__(self, model_name: str = "Falconsai/text_summarization", version: int = 1) -> None:

        self.version = version
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def process(self, text, language):
        
        language_name = {
            "en": "english",
            "pt": "portuguese",
            "de": "german",
            "nl": "dutch",
            "es": "spanish",
            "ru": "russian",
            "pl": "polish",
        }
        
        if self.version == 1:
            summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, truncation=True)
            return summarizer(text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
        elif self.version == 2:
            self.model.config.prefix = "summarize: "
            summarizer = SummarizationPipeline(model=self.model, tokenizer=self.tokenizer, truncation=True)
            return summarizer(text, max_length=512, min_length=30, do_sample=False)[0]['summary_text']
        elif self.version == 3:
            prefix = f"summarize in language {language_name[language]}: "
            print("#####################################")
            print(prefix)
            encoded_input = self.tokenizer.encode(prefix + text, return_tensors='pt', truncation=True)
            with torch.no_grad():
                output = self.model.generate(encoded_input.to(self.device), max_length=encoded_input.shape[1], min_length=0, do_sample=False)
            return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]



 
             
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarizer_model", default="Falconsai/text_summarization", type=str)
    parser.add_argument("--spacy_model", default="en_core_web_sm", type=str)
    parser.add_argument("--data", default="small_subset.csv", type=str)
    args = parser.parse_args()
    
    spacy_model = spacy.load(args.spacy_model)
    summarizer = Summarizer(args.summarizer_model, version=3)
    
    df = pd.read_csv(args.data)
    df = df[df['language'].str.contains('en')]
    
    
    for text in df['text'].tolist():
        print("==============================")
        try:
            print("### Or text: \n", text)
            print("### Summary: \n", summarizer.process(text))
            print("### Subjects: \n", spacy_model(text).ents)
        except Exception as e:
            print(e)
            continue
        
        
        
        
        
        
        
        
        

###

# 1 Sumarization
    # Hugging Face
    
# 2 Keywords
    # Vicuna, instruction based
    # Opt66-iml, huggin face
    # ChatGpt
    # Gemini
    # Falcone, huggin face
    
    
      
# 3 Paraphrase
    # Vicuna, instruction based
    # Opt66-iml, huggin face
    # ChatGpt
    # Gemini
    # Falcone, huggin face