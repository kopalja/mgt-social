import argparse
import os
from enum import Enum

import pandas as pd
from tqdm import tqdm

from gemini import Gemini
from spacy_tagging import spacy_keywords
from summarizer import Summarizer


class GenerationType(Enum):
    paraphrase = "paraphrase"
    k_to_one = "k_to_one"
    summarizer = "summarizer"
    keywords = "keywords"

    def __str__(self):
        return self.value


def spacy(gemini, text: str, language: str):
    print("### Inpt", text)
    keywords = spacy_keywords(text, language)
    prompt = f"Generate sentense in {gemini.language_code_to_name[language]} containing the following words: {', '.join(keywords)}"
    result = gemini.query(prompt)
    return result


def k_to_one(k: int = 6) -> dict:
    data = dict([(n, []) for n in ["output", "language", "source"]])
    groups = df.groupby(["language", "source"])
    for (language, source), group in groups:
        for _ in range(3):
            texts = list(group.sample(n=k)["text"])
            gemini_response = gemini.similar_to_n(texts, language)
            if gemini_response.startswith(f"{k+1}: "):
                gemini_response = gemini_response[3:]
            data["source"].append(source)
            data["language"].append(language)
            data["output"].append(gemini_response)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarizer_model", default="Falconsai/text_summarization", type=str)
    parser.add_argument("--gemini_project_name", default="mgt-social", type=str)
    parser.add_argument("--gemini_model_name", default="gemini-1.0-pro", type=str)
    parser.add_argument("--gemini_location", default="us-central1", type=str)
    parser.add_argument("--data", default="small_subset.csv", type=str)
    parser.add_argument("--languages", default=["en", "pt", "de", "nl", "es", "ru", "pl"], nargs="+")
    parser.add_argument("--type", type=GenerationType, choices=list(GenerationType), default=GenerationType.keywords)
    args = parser.parse_args()

    # 1) Create models
    gemini = Gemini(args.gemini_project_name, args.gemini_location, args.gemini_model_name, debug=True)
    summarizer = Summarizer(args.summarizer_model, version=3)

    # 2) Preprocess data
    df = pd.read_csv(args.data)
    df = df[df["language"].isin(args.languages)]
    df = df[["text", "language", "source"]]

    if args.type == GenerationType.k_to_one:
        pd.DataFrame(data=k_to_one()).to_csv("data/data_k_to_one.csv")
    else:
        data = dict([(n, []) for n in ["input", "output", "language", "source"]])
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
        pd.DataFrame(data=data).to_csv(os.path.join("data", f"data_{args.type}.csv"))
