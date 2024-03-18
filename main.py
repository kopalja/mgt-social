
import argparse
import pandas as pd
from tqdm import tqdm

from spacy_tagging import spacy_keywords
from gemini import Gemini
from summarizer import Summarizer
from enum import Enum



class GenerationType(Enum):
    paraphrase = 'paraphrase'
    k_to_one = 'k_to_one'
    summarizer = 'summarizer'
    keywords = 'keywords'

    def __str__(self):
        return self.value



def spacy(gemini, text: str, language: str):
    words = spacy_keywords(text, language)
    instruction = {
        "en": "Generate sentense containing the following words",
        "pt": "Gere uma frase contendo as seguintes palavras",
        "de": "Bilden Sie einen Satz mit den folgenden Wörtern",
        "nl": "Genereer een zin met de volgende woorden",
        "es": "Genera una oración que contenga las siguientes palabras",
        "ru": "Создайте предложение, содержащее следующие слова",
        "pl": "Utwórz zdanie zawierające następujące słowa"
    }
    result = gemini.query(f"{instruction[language]}: {' '.join(words)}")
    return result


def k_to_one(k: int = 6):
    data = {
        "output": [],
        "language": [],
        "source": []}
                
    groups = df.groupby(['language', 'source'])
    for (language, source), group in groups:
        for _ in range(3):
            texts = list(group.sample(n = k)['text'])
            gemini_response = gemini.similar_to_n(texts, language)
            if gemini_response.startswith(f'{k+1}: '):
                gemini_response = gemini_response[3:]
            data["source"].append(source)
            data["language"].append(language)
            data["output"].append(gemini_response)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarizer_model", default="Falconsai/text_summarization", type=str)
    parser.add_argument("--g_model_name", default="gemini-1.0-pro", type=str)
    parser.add_argument("--g_project_name", default="mgt-social", type=str)
    parser.add_argument("--g_location", default="us-central1", type=str)
    parser.add_argument("--data", default="small_subset.csv", type=str)
    parser.add_argument("--languages", default=["en", "pt", "de", "nl", "es", "ru", "pl"], nargs='+')
    parser.add_argument('--type', type=GenerationType, choices=list(GenerationType), default=GenerationType.k_to_one)
    args = parser.parse_args()


    # 1) Create models
    gemini = Gemini(args.g_project_name, args.g_location, args.g_model_name, debug=False)
    summarizer = Summarizer(args.summarizer_model, version=3)
    
    # 2) Preprocess data
    df = pd.read_csv(args.data)
    df = df[df['language'].isin(args.languages)]
    df = df[['text', 'language', 'source']]
    df = df[df['language'] == 'en']

    if args.type == GenerationType.k_to_one:
        pd.DataFrame(data = k_to_one()).to_csv('data/data_k_to_one.csv')
    else:
        data = {
            "input": [],
            "output": [],
            "language": [],
            "source": []
        }
        for row in tqdm(df.itertuples()):
            data["source"].append(row.source)
            data["language"].append(row.language)
            data["input"].append(row.text)
            
            if args.type == GenerationType.paraphrase:
                data["output"].append(gemini.paraphrase(row.text, row.language))
            elif args.type == GenerationType.keywords:
                data["output"].append(spacy(gemini, row.text, row.language))
            elif args.type == GenerationType.summarizer:
                summ = summarizer.process(row.text, row.language)
                data["output"].append(gemini.paraphrase(summ, row.language, iterations=1))
                print(row.text)
                print(summ)
        pd.DataFrame(data = data).to_csv(f'data/data_{args.type}.csv')
        

        
        