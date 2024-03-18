from typing import List
import spacy
import random

def spacy_keywords(text: str, lang: str, sample_size: int = 2) -> List[str]:
    if lang == "en":
        nlp = spacy.load("en_core_web_sm")
    else:
        nlp = spacy.load(f"{lang}_core_news_sm")
        
    doc = nlp(text)
    keywords = [e.text for e in doc if e.pos_ in ["NOUN", "VERB", "PROPN"]]

    if len(keywords) < sample_size:
        words = [w for w in text.split(' ') if len(w) > 2]
        random.shuffle(words)
        keywords.extend(words[:sample_size - len(keywords)])

    random.shuffle(keywords)
    return keywords[:sample_size]