from enum import Enum
from typing import Union

from .hf_model import HFModel
from .openai_model import OpenAIModel
from .cohere_model import CohereModel
from .hf_tgi_model import HFTGIModel
from .base import BaseModel
from .response import ResponseObject
from .status_tracker import StatusTracker
from .api_model import APIModel


class MODEL_TYPE(str, Enum):
    HF = "HF"
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    TGI = "TGI"


HF_MODELS = [
    "facebook/opt-iml-max-1.3b",
    "facebook/opt-iml-max-30b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-7b-chat-hf",  # a specific format needs to be followed
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "stabilityai/StableBeluga2",  # 70B, uses a special prompt format
    "stabilityai/StableBeluga-7B",  # 13B
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    "google/flan-t5-small",  # 80M
    "google/flan-t5-base",  # 248M
    "google/flan-t5-large",  # 783M
    "google/flan-t5-xl",  # 3b?
    "google/flan-t5-xxl",  # 11.3b
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "Writer/camel-5b-hf",
    "Writer/InstructPalmyra-20b",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b",
]

OPENAI_MODELS = [
    # "gpt-3.5-turbo-0301",  # to be deprecated in June 2024
    # "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",  # latest model version
    # 'text-davinci-003',  # legacy
    # 'gpt-3.5-turbo-instruct',  # legacy replacement for text-davinci-003
    "gpt-4-0613",
]

COHERE_MODELS = [
    "command",  # instruction-tuned
    "command-light",
    "base-light",  # generation
    "base",
]

MODELLIST = HF_MODELS + OPENAI_MODELS + COHERE_MODELS
ModelListEnum = Enum("ModelEnum", {model: model for i, model in enumerate(MODELLIST)})


def model_name_to_str(model_name: str):
    if "/" in model_name:
        return model_name.split("/")[1].replace("-", "")
    else:
        return model_name.replace("-", "")


def get_model(model_type: MODEL_TYPE):
    if model_type == MODEL_TYPE.HF:
        return HFModel
    elif model_type == MODEL_TYPE.OPENAI:
        return OpenAIModel
    elif model_type == MODEL_TYPE.COHERE:
        return CohereModel
    elif model_type == MODEL_TYPE.TGI:
        return HFTGIModel
    else:
        raise ValueError(f"Invalid model type: {model_type}")
