from torch import bfloat16, float16
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


class HFDefaultSettings:
    """Stores default settings for the HF models and tokenizers."""

    use_fast = True
    precision = float16
    pipeline_name = "text-generation"
    model_class = AutoModelForCausalLM
    tokenizer_class = AutoTokenizer
    tokenizer_legacy = False

    def __init__(self, model_name):
        # model specific settings
        if "flan-t5" in model_name:
            self.pipeline_name = "text2text-generation"
            self.use_fast = False
            self.model_class = T5ForConditionalGeneration
            # tokenizer_class = T5Tokenizer
        elif "dolly" in model_name:
            # instantiate pipeline without task to use Dolly's
            #   custom InstructionTextGenerationPipeline
            self.pipeline_name = None
            self.precision = bfloat16
        elif "opt" in model_name:
            # according to the model card, the fast tokenizer is not supported
            #  for opt-iml-max-30b
            self.use_fast = False
        elif "falcon" in model_name:
            self.precision = bfloat16
        elif "stable" in model_name:
            self.use_fast = False
