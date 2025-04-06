import logging
import os
import time
from typing import Tuple

import torch
from torch import bfloat16, float16
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration

from models.base import BaseModel
from models.hf_defaults import HFDefaultSettings
from models.response import ResponseObject

logger = logging.getLogger(__name__)

"""
A class to encapsulate the huggingface pipeline.
should be able to:
- instantiate a model
- create a text-generation pipeline
- run the model/pipeline with a prompt (already formatted)
- takes text-generation arguments
"""


class HFModel(BaseModel):
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.getenv("HUGGINGFACE_HUB_CACHE")

    def __init__(self, model_name, output_dir=None):
        # call super constructor
        super().__init__(model_name, output_dir)

        logger.info(f"Using model: {model_name}")

        is_cuda = torch.cuda.is_available()
        devices = torch.cuda.device_count()

        # list devices
        if is_cuda:
            logger.info(f"GPU available: {is_cuda}")
            logger.info(f"Number of devices: {devices}")
            for i in range(devices):
                logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")

        logger.info(f"HuggingFace cache: {os.environ['HUGGINGFACE_HUB_CACHE']}")
        self.sleep_time = 0

        # default settings
        hf_settings = HFDefaultSettings(model_name)

        logger.info(f"Using {hf_settings.pipeline_name} pipeline for {model_name}")
        logger.info(f"Using precision: {hf_settings.precision}")

        model = hf_settings.model_class.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=hf_settings.precision,
            cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
        )

        self.tokenizer = hf_settings.tokenizer_class.from_pretrained(
            model_name,
            use_fast=hf_settings.use_fast,
            legacy=hf_settings.tokenizer_legacy,
        )

        self.pipeline = pipeline(
            task=hf_settings.pipeline_name,
            model=model,
            tokenizer=self.tokenizer,
        )

    def process_gen_kwargs(self, **kwargs):
        if "flan-t5" in self.model_name:
            kwargs.pop("return_full_text", None)
        elif "dolly" in self.model_name:
            maxlen = kwargs.pop("max_length", None)
            kwargs["max_new_tokens"] = maxlen
        elif "falcon" in self.model_name:
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        elif "Palmyra" in self.model_name:
            kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        return kwargs

    def run(self, sentence_id, prompt, **kwargs) -> ResponseObject:
        """run the pipeline with a prompt (already formatted)"""
        t0 = time.time()

        # optionally override model generation kwargs
        kwargs = self.process_gen_kwargs(**kwargs)

        try:
            retval = self.pipeline(prompt, **self.gen_kwargs)
        except Exception as e:
            logger.error(f"Error running model: {e}")
            raise e

        output = self._extract_gen_text(retval)
        duration = time.time() - t0

        return ResponseObject(
            output=output,
            duration=duration,
            sentence_id=sentence_id,
        )

    def _extract_gen_text(self, retval):
        if "generated_text" in retval:
            return retval["generated_text"].strip()
        else:
            return retval[0]["generated_text"].strip()

    def run_batch(self, prompt_info_batch: list) -> list:
        t0 = time.time()
        results = {}

        max_batch_len = max([k["kwargs"]["max_length"] for k in prompt_info_batch])

        kwargs = prompt_info_batch[0]["kwargs"]
        kwargs["max_length"] = max_batch_len
        kwargs = self.process_gen_kwargs(**kwargs)

        sentences = [prompt_info["prompt"] for prompt_info in prompt_info_batch]

        try:
            retval = self.pipeline(sentences, **kwargs)
        except Exception as e:
            logger.error(f"Error running model: {e}")
            raise e

        duration = (time.time() - t0) / len(prompt_info_batch)

        for i, prompt_info in enumerate(prompt_info_batch):
            sentence_id = prompt_info["sentence_id"]
            output = self._extract_gen_text(retval[i])
            results[sentence_id] = ResponseObject(
                output=output,
                duration=duration,
                sentence_id=sentence_id,
            )

        return results

    def _tokenize(self, text):
        return self.tokenizer(text)

    def count_tokens(self, input_prompt: str, text: str) -> Tuple[int, int]:
        num_prompt_tokens = len(self._tokenize(input_prompt)["input_ids"])
        num_text_tokens = len(self._tokenize(text)["input_ids"])

        return num_prompt_tokens, num_text_tokens

    def update_max_len(self, max_len):
        self.params.update_max_len(max_len)
