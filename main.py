import json
import httpx
from pydantic import BaseModel, Field
from fastapi import FastAPI
from typing import Dict, Any, Optional

app = FastAPI()


class PredictRequest(BaseModel):
    hf_pipeline: str
    model_deployed_url: str
    inputs: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

def text_generation_encoder(inputs, parameters):
    arg = {
        "name": "array_inputs",
        "shape": [1],
        "datatype": "BYTES",
        "data": [inputs],
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

def zero_shot_classification_encoder(inputs, parameters):
    arg1 = {
        "name": "array_inputs",
        "shape": [1],
        "datatype": "BYTES",
        "data": [inputs],
    }
    arg2 = {
        "name": "candidate_labels",
        "shape": [ len(parameters["candidate_labels"])],
        "datatype": "BYTES",
        "parameters": {"content_type": "str"},
        "data": parameters["candidate_labels"],
    }
    input = [arg1, arg2]

    for key in parameters:
        if key != "candidate_labels":
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

def token_classification_encoder(inputs, parameters):
    arg1 = {
        "name": "array_inputs",
        "shape": [1],
        "datatype": "BYTES",
        "data": [inputs],
    }
    input = [arg1]

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

def object_detection_encoder(inputs, parameters):
    arg = {
        "name":"inputs",
        "shape": [1],
        "datatype": "BYTES",
        "data": [inputs],
    }
    if inputs.startswith("http"):
        arg["parameters"] = {"content_type": "str"}
    else:
        arg["parameters"] = {"content_type": "pillow_image"}
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

# Create a dictionary to map strings to functions
encoder_mapping = {
    "text-generation": text_generation_encoder,
    "zero-shot-classification": zero_shot_classification_encoder,
    "token-classification": token_classification_encoder,
    "object-detection": object_detection_encoder,
    # add as many encoder functions as you want
}

def v2_request_body(hf_pipeline, inputs, parameters):
    # Get the corresponding encoder from the dictionary
    encoder = encoder_mapping.get(hf_pipeline)

    if encoder:
        return encoder(inputs, parameters)
    else:
        return "Invalid HF Pipeline"

@app.post(path="/predict")
async def predict(request: PredictRequest):
    # Write your code here to translate input into V2 protocol and send it to model_deployed_url

    url = request.model_deployed_url
    payload = v2_request_body(request.hf_pipeline, request.inputs, request.parameters)

    response = httpx.post(url, data=payload)

    output_list = response.json()["outputs"][0]["data"]
    output_data = json.loads(output_list[0])
    output = [output_data]

    if request.hf_pipeline == "token-classification":
        return output[0]
    else :
        return output