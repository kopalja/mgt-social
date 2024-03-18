import argparse
import time
from typing import List

import pandas as pd
import vertexai
from vertexai.generative_models import (GenerationConfig, GenerativeModel,
                                        HarmBlockThreshold, HarmCategory)


class Gemini:
    def __init__(self, project_id: str, location: str, model_name: str, debug: bool = False) -> None:
        self.debug = debug
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        self.config = GenerationConfig(
            temperature=0.0,
            # top_k=32,
            # top_p=0.8,
            candidate_count=1,
            max_output_tokens=2048,
        )
        self.safety_config = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        }
        self.language_code_to_name = {
            "en": "english",
            "pt": "portuguese",
            "de": "german",
            "nl": "dutch",
            "es": "spanish",
            "ru": "russian",
            "pl": "polish",
        }

    def query(self, inpt: str, sleep: int = 2) -> str:
        try:
            if self.debug:
                print(f"### Prompt:\n{inpt}")
            response = self.model.generate_content(
                inpt, generation_config=self.config, safety_settings=self.safety_config
            ).text
            if self.debug:
                print(f"### Response:\n{response}")
            time.sleep(sleep)
            return response
        except Exception as e:
            print(f"Error when asking gemini: {e}")
            return "No response from Gemini."

    def paraphrase(self, text: str, language: str, iterations: int = 3):
        instruction = f"Paraphrase text in {self.language_code_to_name[language]}."
        for _ in range(iterations):
            text = self.query(f"{instruction} Text to paraphrase: {text}")
        return text

    def similar_to_n(self, texts: List[str], language: str):
        intro = f"Here are {len(texts)} short texts in language {self.language_code_to_name[language]} labeled from 1 to {len(texts)} each ended with ###."
        # ending = "Create one new social media post similar to previous ones in structure and length"
        # ending = "Create one new social media post that should be indistinguishable from the showed examples."
        ending = "Create one new text that should be indistinguishable from the previous examples."
        middle = "\n".join([f"{i+1}: {t} #" for i, t in enumerate(texts)])
        prompt = f"{intro}:\n{middle}\n{ending}"
        return self.query(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gemini-1.0-pro", type=str)
    parser.add_argument("--project_name", default="mgt-social", type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--data", default="small_subset.csv", type=str)
    args = parser.parse_args()

    model = Gemini(args.project_name, args.location, args.model_name)
    df = pd.read_csv(args.data)
    df = df[df["language"].str.contains("en")]

    for c in df["text"].tolist():
        print("==============================")
        print(c)
        print("###")
        print(model.paraphrase(c))
