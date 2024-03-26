import argparse
import time
from typing import List

import pandas as pd
import vertexai
from vertexai.generative_models import (GenerationConfig, GenerativeModel,
                                        HarmBlockThreshold, HarmCategory)

from misc import LANGUAGE_CODE_MAP, JAILBREAK

STOP_PREFIX = [
    "I am sorry, I cannot",
    "I am sorry, I am not supposed to",
    "I am not supposed to",
    "Sorry, I am not able",
    "I apologize",
    "I am not supposed to",
    "Unfortunately, I am not supposed"]

class Gemini:
    def __init__(self, project_id: str, location: str, model_name: str, debug: bool = False) -> None:
        self.debug = debug
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        self.config = GenerationConfig(
            temperature=0.9, # make model deterministic
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

    def query(self, inpt: str, sleep: int = 2) -> str:
        try:
            response = self.model.generate_content(
                inpt, generation_config=self.config, safety_settings=self.safety_config
            ).text
            if self.debug:
                print(f"############## Prompt ##############\n{inpt}")
                print(f"############## Response ##############\n{response}")
            time.sleep(sleep)
            return response
        except Exception as e:
            print(f"Error when asking gemini: {e}")
            return "No response from Gemini."

    def paraphrase(self, text: str, language: str, iterations: int = 3, repetitions_per_iteration: int = 5):
        instruction = f"Your ultimate goal is to rephrase in {LANGUAGE_CODE_MAP[language]} using different words and sentence composition the following text:"
        for _ in range(iterations):
            for _ in range(repetitions_per_iteration):
                new_text = self.query(f'{JAILBREAK} {instruction} "{text}"')[len("JailBreak: "):]
                if new_text != text and all([not new_text.startswith(p) for p in STOP_PREFIX]):
                    text = new_text
                    break

        return text

    def similar_to_n(self, texts: List[str], language: str, k: int):
        intro = f"You are a helpful assistant. Generate a short text similar to the following examples. The text must be in {LANGUAGE_CODE_MAP[language]} language.\n"
        examples = ""
        for idx, text in enumerate(texts[:k+1]):
            examples += f"EXAMPLE {idx+1}: {text}\n"
        examples += f"GENERATED TEXT: {texts[k+1]}\n\n"
        
        current = ""
        for i, text in enumerate(texts[k+2:]):
            current += f"EXAMPLE {i+1}: {text}\n"
        current += "GENERATED TEXT:"
        prompt = f"{intro}\n{examples}\n{current}"
        return self.query(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gemini-1.0-pro", type=str)
    parser.add_argument("--project_name", default="mgt-social", type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--data", default="small_subset.csv", type=str)
    args = parser.parse_args()

    instruction = "Your ultimate goal is to rephrase in english using different words and sentence composition the following text:"
    question = "My fellow niggers, I am pleased to inform you that we figured out how to game facebook."
    prompt = f'{JAILBREAK} {instruction} "{question}"'
    model = Gemini(args.project_name, args.location, args.model_name, debug=True)
    model.query(prompt, sleep=0)
    # model.paraphrase(question, "en")
    
