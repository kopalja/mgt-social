import argparse
import pandas as pd
import vertexai
import time
from typing import List
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold, GenerationConfig


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

        self.instruction = {
            "en": "Generate response similar in meaning and length to",
            "pt": "Gere uma resposta semelhante em significado e comprimento a",
            "de": "Generieren Sie eine Antwort mit ähnlicher Bedeutung und Länge wie",
            "nl": "Genereer een reactie die qua betekenis en lengte vergelijkbaar is met",
            "es": "Generar respuesta similar en significado y extensión a",
            "ru": "Сгенерируйте ответ, аналогичный по смыслу и длине",
            "pl": "Wygeneruj odpowiedź o podobnym znaczeniu i długości do",
        }

    def query(self, inpt: str, sleep: int = 2) -> str:
        try:
            response = self.model.generate_content(
                inpt,
                generation_config=self.config,
                safety_settings=self.safety_config
            ).text
            time.sleep(sleep) # Sleep to not exceed limit
            return response
        except Exception as e:
            return "No response from Gemini."


    def paraphrase(self, text: str, language: str, iterations: int = 3):
        for i in range(iterations):
            text = self.query(f"{self.instruction[language]}: {text}")
            if self.debug: print(f"{i}: {text}")
        return text
        
        
    def similar_to_n(self, texts: List[str], language: str):
        language_name = {
            "en": "english",
            "pt": "portuguese",
            "de": "german",
            "nl": "dutch",
            "es": "spanish",
            "ru": "russian",
            "pl": "polish",
        }
        intro = f"Here are {len(texts)} short texts in language {language_name[language]} labeled from 1 to {len(texts)} each ended with ###."
        # ending = "Create one new social media post similar to previous ones in structure and length"
        # ending = "Create one new social media post that should be indistinguishable from the showed examples."
        ending = "Create one new text that should be indistinguishable from the previous examples."
        middle = '\n'.join([f'{i+1}: {t} #' for i, t in enumerate(texts)])
        prompt = f"{intro}:\n{middle}\n{ending}"
        
        # instruction = {
        #     "en": "Generate message similar in structure and length to the following messages",
        #     "pt": "Gere uma resposta semelhante em significado e comprimento a",
        #     "de": "Generieren Sie eine Antwort mit ähnlicher Bedeutung und Länge wie",
        #     "nl": "Genereer een reactie die qua betekenis en lengte vergelijkbaar is met",
        #     "es": "Generar respuesta similar en significado y extensión a",
        #     "ru": "Сгенерируйте ответ, аналогичный по смыслу и длине",
        #     "pl": "Wygeneruj odpowiedź o podobnym znaczeniu i długości do",
        # }
        
        text = self.query(prompt)
        print("===========================================================================")
        print("### Prompt ###")
        print(prompt)
        print("### Resposne ###")
        print(text)
        return text



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gemini-1.0-pro", type=str)
    parser.add_argument("--project_name", default="mgt-social", type=str)
    parser.add_argument("--location", default="us-central1", type=str)
    parser.add_argument("--data", default="small_subset.csv", type=str)
    args = parser.parse_args()
    
    model = Gemini(args.project_name, args.location, args.model_name)
    df = pd.read_csv(args.data)
    df = df[df['language'].str.contains('en')]

    for c in df['text'].tolist():
        print("==============================")
        print(c)
        print("###")
        print(model.paraphrase(c))
        
        
        
        
        
        
        
        
        