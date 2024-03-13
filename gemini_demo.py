import vertexai
from vertexai.generative_models import GenerativeModel, Part
import time


def generate_text(project_id: str, location: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    multimodal_model = GenerativeModel("gemini-1.0-pro")
    # Query the model
    response = multimodal_model.generate_content("Generate sentence similar to: hey levi u wanna play league of legend")
    print(response)
    print(response.text)
    return response.text





def init_model(project_id: str, location: str) -> str:
    vertexai.init(project=project_id, location=location)
    return GenerativeModel("gemini-1.0-pro")

def predict(model, inpt: str) -> str:
    response = model.generate_content(f"Generate sentence similar to: {inpt}")
    print(response)
    return response.text

def paraphrase(model, text: str, iterations: int = 5):
    for i in range(iterations):
        text = predict(model, text)
        print(text)
        time.sleep(3)
    return text

if __name__ == '__main__':
    model = init_model("mgt-social", "europe-west4")
    # text = "hey levi u wanna play league of legend"
    text = "i need to go back to duoling"
    
    r = paraphrase(model, text)
    print(r)
