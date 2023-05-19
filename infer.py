import json
import requests
from urllib.parse import urljoin
from fastapi import FastAPI


# https://test1-intern-divya.demo1.truefoundry.com
# if above is the deployed url, then extract_model_name should return test1
def extract_model_name(model_deployed_url):
    return model_deployed_url.split("-")[0].split("//")[1]


def text_generation_encoder(inputs, parameters):
    arg = {
        "name": "array_inputs",
        "shape": [1],
        "datatype": "BYTES",
        "data": [str(inputs)],
    }
    input = [arg]
    for key in parameters:
        param = {
            "name": key,
            "shape": [-1],
            "datatype": "BYTES",
            "data": [str(parameters[key])],
            "parameters": {"content_type": "hg_json"},
        }
        input.append(param)

    payload = json.dumps({"inputs": input})
    return payload


def zero_shot_classification_encoder():
    print("Function Two called")


def token_classification_encoder():
    print("Function Three called")


def object_detection_encoder():
    print("Function Four called")


# Create a dictionary to map strings to functions
encoder_mapping = {
    "text-generation": text_generation_encoder,
    "zero-shot-classification": zero_shot_classification_encoder,
    "token-classification": token_classification_encoder,
    "object-detection": object_detection_encoder,
}


def v2_request_body(hf_pipeline, inputs, parameters):
    # Get the corresponding encoder from the dictionary
    encoder = encoder_mapping.get(hf_pipeline)

    # Check if the encoder exists in the mapping
    if encoder:
        return encoder(inputs, parameters)
    else:
        return "Invalid HF Pipeline"


app = FastAPI()


@app.post("/predict")
async def predict(hf_pipeline: str, model_deployed_url: str, inputs, parameters: dict):
    ENDPOINT_URL = model_deployed_url
    model_name = extract_model_name(model_deployed_url)
    url = urljoin(ENDPOINT_URL, f"v2/models/{model_name}/infer")

    headers = {"Content-Type": "application/json"}

    payload = v2_request_body(hf_pipeline, inputs, parameters)

    print(payload)

    response = requests.post(url, headers=headers, data=payload)
    return response.json()
