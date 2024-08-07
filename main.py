import argparse
import os
from enum import Enum

import pandas as pd

from models.aya import Aya
from models.eagle import Eagle
from models.falcon import Falcon
from models.gemini import Gemini
from models.mistral import Mistral
from models.opt import Opt
from models.vicuna import Vicuna
from spacy_tagging import long_words, spacy_keywords
from summarizer import Summarizer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class GenerationType(Enum):
    paraphrase = "paraphrase"
    k_to_one = "k_to_one"
    summarizer = "summarizer"
    keywords = "keywords"

    def __str__(self):
        return self.value


def k_to_one(model, data: pd.DataFrame, k: int = 6, examples_per_group: int = 3) -> dict:
    results = {col: [] for col in ["input", "output", "source", "language"]}
    groups = data.groupby(["language", "source"])
    for (language, source), group in groups:
        for _ in range(examples_per_group):
            # We need 2k+1 number of samples - k+1 for the in-context example and k for the generation
            if len(group) < 2 * k + 1:
                print(
                    f"Not enough samples for language {language} and source {source}. At least {2*k+1} are needed but there are only {len(group)}. Skipping..."
                )
                break
            texts = list(group.sample(n=2 * k + 1)["text"])
            results["input"].append("\n".join(texts))
            results["source"].append(source)
            results["language"].append(language)
            results["output"].append(model.similar_to_n(texts, language, k))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="small_subset.csv", type=str)
    parser.add_argument("--output_root_dir", default="output", type=str)
    parser.add_argument("--summarizer_model", default="Falconsai/text_summarization", type=str)
    parser.add_argument("--gemini_project_name", default="mgt-social", type=str)
    parser.add_argument("--gemini_model_name", default="gemini-1.0-pro", type=str)
    parser.add_argument("--gemini_location", default="us-central1", type=str)
    parser.add_argument("--vicuna_path", default="/mnt/dominik.macko/vicuna-13b", type=str)
    parser.add_argument(
        "--mistral_path",
        default="/mnt/jakub.kopal/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/73068f3702d050a2fd5aa2ca1e612e5036429398",
        type=str,
    )
    parser.add_argument("--eagle_path", default="/mnt/michal.spiegel/models/eagle-7b", type=str)
    parser.add_argument("--opt_path", default=None, type=str)
    parser.add_argument("--aya_path", type=str)
    parser.add_argument("--falcon_path", default=None, type=str)
    parser.add_argument(
        "--languages", default=["en", "pt", "de", "nl", "es", "ru", "pl", "ar", "bg", "ca", "uk", "pl", "ro"], nargs="+"
    )
    parser.add_argument("--type", type=GenerationType, choices=list(GenerationType), default=GenerationType.keywords)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=".cache")
    args = parser.parse_args()

    # 1) Create models
    summarizer = Summarizer(args.summarizer_model, version=3)
    if args.model_name == "gemini":
        model = Gemini(args.gemini_project_name, args.gemini_location, args.gemini_model_name, debug=True)
    elif args.model_name == "vicuna":
        model = Vicuna(args.vicuna_path, debug=True, use_gpu=True, cache_dir=args.cache_dir)
    elif args.model_name == "mistral":
        model = Mistral(args.mistral_path, debug=True, use_gpu=True, cache_dir=args.cache_dir)
    elif args.model_name == "eagle":
        model = Eagle(args.eagle_path, debug=True, use_gpu=True, cache_dir=args.cache_dir)
    elif args.model_name == "opt":
        model = Opt(args.opt_path, debug=True, use_gpu=True, cache_dir=args.cache_dir)
    elif args.model_name == "falcon":
        model = Falcon(args.falcon_path, debug=True, use_gpu=True, cache_dir=args.cache_dir)
    elif args.model_name == "aya":
        model = Aya(args.aya_path, debug=True, use_gpu=True, cache_dir=args.cache_dir)
    else:
        raise Exception(
            f"Unsupported model type: {args.model_name}. Supported model names are: `gemini`, `vicuna`, `mistral`, `eagle`, `opt`."
        )

    # 2) Preprocess data
    df = pd.read_csv(args.data)
    df = df[df["language"].isin(args.languages)]
    df = df[["text", "language", "source"]]

    output_dir = os.path.join(args.output_root_dir, args.model_name)
    os.makedirs(args.output_root_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if args.type == GenerationType.k_to_one:
        pd.DataFrame(k_to_one(model, df, k=3)).to_csv(
            os.path.join(output_dir, f"{args.model_name}_k_to_one.csv"), index=False
        )
    else:
        data = dict([(n, []) for n in ["input", "output", "language", "source"]])
        for row in df.itertuples():
            data["source"].append(row.source)
            data["language"].append(row.language)
            data["input"].append(row.text)

            if args.type == GenerationType.paraphrase:
                print("paraphrasing")
                data["output"].append(model.paraphrase(row.text, row.language))
            elif args.type == GenerationType.keywords:
                data["output"].append(model.keywords(long_words(row.text), row.language))
            elif args.type == GenerationType.summarizer:
                summ = summarizer.process(row.text, row.language)
                data["output"].append(model.paraphrase(summ, row.language, iterations=1))
        pd.DataFrame(data=data).to_csv(os.path.join(output_dir, f"{args.model_name}_{args.type}.csv"), index=False)
