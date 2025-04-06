import os
from typing import Tuple
import asyncio
import time
import logging

from huggingface_hub import InferenceClient, AsyncInferenceClient
from transformers import AutoTokenizer, T5Tokenizer
import json
import jsonlines

from models.api_request import APIRequest
from models.api_model import APIModel
from models.response import ResponseObject
from models.status_tracker import StatusTracker
from models.hf_defaults import HFDefaultSettings

logger = logging.getLogger(__name__)


class TGIRequest(APIRequest):
    async def _call_api(
        self,
        status_tracker: StatusTracker,
    ):
        await asyncio.sleep(0.01)
        error = None
        response = None
        try:
            # call api
            logger.debug(f"Calling API for sentence {self.sentence_id}.")
            async with asyncio.timeout(45):
                response = await self.client.text_generation(
                    prompt=self.prompt,
                    details=True,
                    **self.kwargs,
                )
            logger.debug(f"API call complete for sentence {self.sentence_id}.")
        except TimeoutError as e:
            error = f"Timeout error on sentence {self.sentence_id}: {e}"
            logger.warning(error)
            status_tracker.num_api_errors += 1
        except Exception as e:
            error = f"Error on sentence {self.sentence_id}: {e}"
            logger.warning(error)
            status_tracker.num_api_errors += 1

            if "rate limit" in error:
                status_tracker.num_rate_limit_errors += 1
                status_tracker.time_of_last_rate_limit_error = time.time()
            elif "blocked output" in error:
                # lowercase the prompt and try again
                self.prompt = self.prompt.lower()

        return (error, response)


class HFTGIModel(APIModel):
    """
    A class representing a High Frequency Text Generation Inference Model.

    Attributes:
    -----------
    model_name : str
        The name of the model.
    output_dir : str, optional
        The output directory for the model, by default None.

    Methods:
    --------
    log_response_info(response)
        Logs the response information.
    num_tokens_from_prompt(prompt)
        Returns the number of tokens from the prompt.
    async_num_tokens_from_prompt(prompt)
        Returns the number of tokens from the prompt asynchronously.
    run_single(sentence_id, prompt, **kwargs)
        Runs a single sentence.
    run_batch(prompt_info_batch)
        Runs a batch of sentences.
    async_tokenize(text)
        Tokenizes the text asynchronously.
    tokenize(text)
        Tokenizes the text.
    async_count_tokens(input_prompt, text)
        Counts the number of tokens in the input prompt and text asynchronously.
    count_tokens(input_prompt, text)
        Counts the number of tokens in the input prompt and text.
    """

    def __init__(self, model_name, output_dir=None, endpoint="https://localhost", port=8080):
        # call super constructor
        super().__init__(model_name, output_dir)

        self.client = InferenceClient(model=f'{endpoint}:{port}')
        self.async_client = AsyncInferenceClient(model=f'{endpoint}:{port}')

        self.log_file = None
        if output_dir is not None:
            self.log_file = os.path.join(output_dir, "log.jsonl")
            self.model_file = os.path.join(output_dir, "model_info.json")
        self.sleep_time = 0.2

        self.template = None
        self.template_len = None

        # default settings
        hf_settings = HFDefaultSettings(model_name)

        self.tokenizer = hf_settings.tokenizer_class.from_pretrained(
            model_name,
            use_fast=hf_settings.use_fast,
            legacy=hf_settings.tokenizer_legacy,
        )

        # async overrides
        self.api_request_cls = TGIRequest

        # log model info
        model_info = {
            "name": self.model_name,
            # "version": self.client.api_version,
        }
        with open(self.model_file, "w") as f:
            json.dump(model_info, f)

    def log_response_info(self, response):
        if self.log_file is None:
            return

        log_json = {
            "finish_reason": response.details.finish_reason,
            "len": response.details.generated_tokens,
            "seed": response.details.seed,
        }
        with jsonlines.open(self.log_file, "a") as f:
            f.write(log_json)

    def run_single(self, sentence_id, prompt, **kwargs):
        """ """
        t0 = time.time()
        error = None
        response = None
        try:
            response = self.client.text_generation(
                prompt=prompt,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error on prompt: {prompt}")
            error = [f"Error on prompt: {prompt}", e]

        duration = time.time() - t0

        response_object = ResponseObject(
            sentence_id=sentence_id,
            output=response,
            errors=error,
            duration=duration,
        )

        return response_object

    def process_gen_kwargs(self, **kwargs):
        # rename max_length to max_new_tokens
        if "max_length" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_length")

        return kwargs

    def run_batch(self, prompt_info_batch: list):
        for prompt_info in prompt_info_batch:
            prompt_info["kwargs"] = self.process_gen_kwargs(**prompt_info["kwargs"])

        results = asyncio.run(self._run_batch(prompt_info_batch))

        for sentence_id, response in results.items():
            if response.output:
                self.log_response_info(response.output)

                response.output = response.output.generated_text.strip()

        return results

    async def async_tokenize(self, text):
        return await self.async_client.tokenize(text)

    def _tokenize(self, text):
        return self.tokenizer(text)

    def tokenize(self, text):
        return self._tokenize(text)

    def num_tokens_from_prompt(self, prompt):
        return len(self._tokenize(prompt)["input_ids"])

    def count_tokens(self, input_prompt: str, text: str) -> Tuple[int, int]:
        if self.template == input_prompt:
            num_prompt_tokens = self.template_len
        else:
            num_prompt_tokens = self.num_tokens_from_prompt(input_prompt)
            self.template = input_prompt
            self.template_len = num_prompt_tokens

        num_text_tokens = self.num_tokens_from_prompt(text)

        return num_prompt_tokens, num_text_tokens
