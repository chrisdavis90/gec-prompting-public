from models import MODEL_TYPE
from prompts import PromptManager, OpenAIPromptManager


def get_prompt_manager(model_type: MODEL_TYPE):
    if model_type == MODEL_TYPE.HF:
        return PromptManager
    elif model_type == MODEL_TYPE.OPENAI:
        return OpenAIPromptManager
    elif model_type == MODEL_TYPE.COHERE:
        return PromptManager
    elif model_type == MODEL_TYPE.TGI:
        return PromptManager
    else:
        raise ValueError(f"Invalid model type: {model_type}")
