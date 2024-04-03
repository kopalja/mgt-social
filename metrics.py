import argparse
import os
import shutil

import editdistance
import ngram
import nltk
import numpy as np
import pandas as pd
import regex
import torch
from nltk import word_tokenize
from nltk.translate import meteor

nltk.download("punkt")
nltk.download("wordnet")

import tensorflow_hub as hub
import tensorflow_text
from evaluate import load
from ftlangdetect import detect
from mauve.compute_mauve import compute_mauve
from polyglot.text import Text, Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def remove_bad_chars(text):
    RE_BAD_CHARS = regex.compile(r"[\p{Cc}\p{Cs}]+")
    return RE_BAD_CHARS.sub("", text)


def get_use_cosine_similarity(subset):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    results = []
    for row in subset.itertuples():
        results.append(cosine_similarity(embed(row.input), embed(row.output), dense_output=False)[0][0])
    return results


def get_bertscore(subset):
    bertscore, results = load("bertscore"), []
    for row in subset.itertuples():
        f1_scores = bertscore.compute(
            predictions=[row.output], references=[row.input], model_type="bert-base-multilingual-cased"
        )
        results.append(sum(f1_scores["f1"]) / len(f1_scores["f1"]))
    return results


def get_meteor(dataset):
    return [meteor([word_tokenize(r.input)], word_tokenize(r.output)) for r in dataset.itertuples()]


def fasttext_detect_language(dataset):
    return [detect(text=r.output.replace("\n", " "), low_memory=False)["lang"] for r in dataset.itertuples()]


def get_ngram(dataset, n=3):
    results = []
    for row in dataset.itertuples():
        try:
            results.append(ngram.NGram.compare(row.input, row.output, N=n))
        except:
            results.append(0)
    return results


def get_tf_cosine_similarity(dataset):

    def custom_tokenizer(text):
        return list(Text(text).words)

    results = []
    for row in dataset.itertuples():
        original = row.input
        obfuscated = row.output
        original_tokens = custom_tokenizer(original)
        try:
            obfuscated_tokens = custom_tokenizer(obfuscated)
            words = set(original_tokens).union(set(obfuscated_tokens))
            vectorizer = CountVectorizer(tokenizer=custom_tokenizer, vocabulary=words)
            original_vector = vectorizer.transform([original])
            obfuscated_vector = vectorizer.transform([obfuscated])
            results.append(round(cosine_similarity(original_vector.toarray(), obfuscated_vector.toarray())[0][0], 4))
        except:
            results.append(0)
    return results


def get_editdistance(dataset):
    return [editdistance.eval(r.input, r.output) for r in dataset.itertuples()]


def get_diff_charlen(dataset):
    prev = ""
    text_charlength = [
        len("".join([y if (y != prev) & ((prev := y) == y) else "" for y in x]).strip()) for x in dataset["input"]
    ]
    generated_charlength = [
        len("".join([y if (y != prev) & ((prev := y) == y) else "" for y in x]).strip()) for x in dataset["output"]
    ]
    generated_charlength_inv = np.array([1 / i if i != 0 else 0 for i in generated_charlength])
    result = text_charlength * generated_charlength_inv
    return np.array([1 / i if i != 0 else 0 for i in result])


def compute_stats_for_model(path: str):
    df = pd.read_csv(path)
    df["model_name"] = [path.split("/")[-1][:-4]] * len(df)
    df['output'] = df['output'].fillna('')

    # if args.recompute:
    #     df['output'] = df['output'].apply(remove_bad_chars)
    if not "mauve" in df or args.recompute:
        df["mauve"] = compute_mauve(
            p_text=df["input"],
            q_text=df["output"],
            featurize_model_name="google-bert/bert-base-multilingual-cased",
            device_id=device,
            max_text_length=512,
            verbose=False,
        ).mauve
    if not "meteor" in df or args.recompute:
        df["meteor"] = get_meteor(df)
    if not "bertscore" in df or args.recompute:
        df["bertscore"] = get_bertscore(df)
    if not "use" in df or args.recompute:
        df["use"] = get_use_cosine_similarity(df)
    if not "fasttext" in df or args.recompute:
        df["fasttext"] = fasttext_detect_language(df)
    if not "ngram" in df or args.recompute:
        df["ngram"] = get_ngram(df, n=3)
    # if not 'tf' in df or args.recompute:
    #     df['tf'] = get_tf_cosine_similarity(df) # Problem with UTF-8 encoding
    if not "editdistance" in df or args.recompute:
        df["editdistance"] = get_editdistance(df)
    if not "ED-norm" in df or args.recompute:
        df["ED-norm"] = df["editdistance"] / [len(x) for x in df["input"]]
    if not "diff_charlen" in df or args.recompute:
        df["diff_charlen"] = get_diff_charlen(df)
    if not "changed_language" in df or args.recompute:
        df["changed_language"] = df["language"] != df["fasttext"]
    if not "LangCheck" in df or args.recompute:
        df["LangCheck"] = len(df[df["changed_language"]]) / len(df)

    df.to_csv(path, index=False)
    return df


def analyzer1(df, grop_by_key: str):
    metrics = ["mauve", "meteor", "bertscore", "ngram", "ED-norm", "diff_charlen", "LangCheck"]
    temp = df.groupby([grop_by_key])[metrics].agg(["mean", "std"])
    for col in temp.columns.get_level_values(0).unique():
        if (col != "mauve") and (col != "LangCheck"):
            temp[col] = [
                f"{str('%.3f' % x)} (±{str('%.2f' % y)})" for x, y in zip(temp[(col, "mean")], temp[(col, "std")])
            ]
        elif col == "LangCheck":
            temp[col] = [f"{str('%.2f' % (x*100))}%" for x, y in zip(temp[(col, "mean")], temp[(col, "std")])]
        else:
            temp[col] = [f"{str('%.3f' % x)}" for x, y in zip(temp[(col, "mean")], temp[(col, "std")])]
    temp.columns = temp.columns.droplevel(level=1)
    temp = temp.T.drop_duplicates().T
    temp.style.highlight_max(props="font-weight: bold;", axis=0).to_html(
        os.path.join(args.output, f"{grop_by_key}.html")
    )


def analyzer2(df):
    temp = pd.DataFrame()
    for model in df["model_name"].unique():
        for lang in ["en", "pt", "de", "nl", "es", "ru", "pl", "ar", "bg", "ca", "uk", "pl", "ro"]:
            dataset = df[(df["model_name"] == model) & (df["language"] == lang)]
            if len(dataset) == 0:
                continue
            langcheck = len(dataset[dataset.language != dataset.fasttext]) / len(dataset)
            temp = pd.concat(
                [
                    temp,
                    pd.DataFrame(
                        {"model": model, "lang": lang, "langcheck": langcheck, "bertscore": dataset.bertscore.mean()},
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    temp2 = (
        temp.pivot_table("langcheck", "model", "lang")
        .style.apply(
            lambda x: [
                "background-color: orange" if v >= 0.8 else "background-color: yellow" if v >= 0.5 else "" for v in x
            ],
            axis=1,
        )
        .format(precision=1)
    )
    temp2.to_html(os.path.join(args.output, "lang_check.html"), table_conversion="matplotlib")

    metrics = ["mauve", "meteor", "bertscore", "ngram", "ED-norm", "diff_charlen", "LangCheck"]

    df.groupby(["model_name", "language"])[metrics].agg(["mean", "std"]).style.format(
        na_rep=0, precision=4
    ).highlight_max(props="font-weight: bold;", axis=0).to_html(os.path.join(args.output, "all.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--output", default="analyzer_output", type=str)
    parser.add_argument("--combined_path", default="combined_data.csv", type=str)
    parser.add_argument("--recompute", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "CPU"
    if os.path.exists(args.combined_path):
        combined_df = pd.read_csv(args.combined_path)
    else:
        combined_df = pd.DataFrame()

    if len(combined_df) == 0 or args.recompute:
        for root, _, files in os.walk(args.root):
            for file in files:
                if file.endswith(".csv"):
                    print(f"Processing: {os.path.join(root, file)}")
                    combined_df = pd.concat(
                        [combined_df, compute_stats_for_model(os.path.join(root, file))], ignore_index=True, copy=False
                    )
        combined_df.to_csv(args.combined_path, index=False)

    analyzer1(combined_df, "model_name")
    analyzer1(combined_df, "language")
    analyzer2(combined_df)
