import random
import re
from typing import List

import spacy

def long_words(text: str, words_to_return: int = 2) -> List[str]:
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = list(set(text.split()))
    words = list(filter(lambda w: (not "http" in w), words))
    words.sort(key = lambda w: -1 * len(w))
    words.extend(["random" for _ in range(2 - len(words))]) # Make sure list has at least two items
    return words[:words_to_return]

def spacy_keywords(text: str, lang: str, sample_size: int = 2) -> List[str]:
    if lang == "en":
        nlp = spacy.load("en_core_web_sm")
    else:
        nlp = spacy.load(f"{lang}_core_news_sm")

    doc = nlp(text)
    keywords = [e.text for e in doc if e.pos_ in ["NOUN", "VERB", "PROPN"]]

    if len(keywords) < sample_size:
        words = [w for w in text.split(" ") if len(w) > 2]
        random.shuffle(words)
        keywords.extend(words[: sample_size - len(keywords)])

    random.shuffle(keywords)
    return keywords[:sample_size]
