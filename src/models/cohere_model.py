import os
from typing import Tuple
import asyncio
import time
import logging

import cohere
import json
import jsonlines

from models.api_request import APIRequest
from models.api_model import APIModel
from models.response import ResponseObject
from models.status_tracker import StatusTracker

logger = logging.getLogger(__name__)

"""
A class to encapsulate the Cohere API.
"""


class CohereRequest(APIRequest):
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
                response = await self.client.generate(
                    model=self.model_name,
                    prompt=self.prompt,
                    **self.kwargs,
                )
            logger.debug(f"API call complete for sentence {self.sentence_id}.")
        except TimeoutError as e:
            error = f"Timeout error on sentence {self.sentence_id}: {e}"
            logger.warning(error)
            status_tracker.num_api_errors += 1
        except Exception as e:
            if "xxx" in self.prompt:
                # hack to get around a single sentence in FCE-dev
                #  that cohere blocks
                error = None
                fake_gen = {
                    "text": "xxx",
                    "id": -1,
                    "finish_reasion": "manual",
                }
                response = cohere.responses.Generations.from_dict(
                    response={
                        "generations": [fake_gen],
                        "prompt": self.prompt,
                    },
                    return_likelihoods="NONE",
                )
                # response = cohere.responses.Generation(
                #     text=self.prompt,
                #     likelihood=0.0,
                #     token_likelihoods=[],
                # )
            else:
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


class CohereModel(APIModel):
    def __init__(self, model_name, output_dir=None):
        # call super constructor
        super().__init__(model_name, output_dir)

        self.client = cohere.Client(os.getenv("COHERE_API_PROD_KEY"))
        self.async_client = cohere.AsyncClient(os.getenv("COHERE_API_PROD_KEY"))

        self.log_file = None
        if output_dir is not None:
            self.log_file = os.path.join(output_dir, "log.jsonl")
            self.model_file = os.path.join(output_dir, "model_info.json")
        self.sleep_time = 0.5

        self.template = None
        self.template_len = None

        # async overrides
        self.api_request_cls = CohereRequest

        # log model info
        model_info = {
            "name": self.model_name,
            "version": self.client.api_version,
        }
        with open(self.model_file, "w") as f:
            json.dump(model_info, f)

    def log_response_info(self, response):
        if self.log_file is None:
            return

        response = response.generations[0]
        token_likelihoods = []
        if response.token_likelihoods is not None:
            token_likelihoods = [
                (tkn_likelihood.token, tkn_likelihood.likelihood)
                for tkn_likelihood in response.token_likelihoods
            ]
        log_json = {
            # "len": len(response.split(" ")),
            "likelihood": response.likelihood,
            "len": max(0, len(token_likelihoods) - 2),
        }
        with jsonlines.open(self.log_file, "a") as f:
            f.write(log_json)

    def num_tokens_from_prompt(self, prompt):
        return len(self.tokenize(prompt))

    async def async_num_tokens_from_prompt(self, prompt):
        return len(await self.async_tokenize(prompt))

    def run_single(self, sentence_id, prompt, **kwargs):
        """Cohere response is:
        #     {
        #   "id": "9c287d79-8e36-42c4-94e3-674c6d382285",
        #   "generations": [
        #     {
        #       "id": "01bea428-5ccf-49a9-9901-abbf34bfa7ed",
        #       "text": "\nLLMs are large language models that have been trained on a massive amount of text data. They"
        #     }
        #   ],
        #   "prompt": "Please explain to me how LLMs work",
        #   "meta": {
        #     "api_version": {
        #       "version": "1"
        #     }
        #   }
        # }
        """
        t0 = time.time()
        error = None
        response = None
        try:
            response = self.client.generate(
                model=self.model_name, prompt=prompt, **kwargs
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

    def run_batch(self, prompt_info_batch: list):
        self.async_client = cohere.AsyncClient(os.getenv("COHERE_API_PROD_KEY"))
        results = asyncio.run(self._run_batch(prompt_info_batch))

        for sentence_id, response in results.items():
            if response.output:
                self.log_response_info(response.output)

                response.output = response.output.generations[0].text.strip()

        return results

    async def async_tokenize(self, text):
        return await self.async_client.tokenize(text)

    def tokenize(self, text):
        return self.client.tokenize(text)

    async def async_count_tokens(self, input_prompt: str, text: str) -> Tuple[int, int]:
        num_prompt_tokens = await self.async_num_tokens_from_prompt(input_prompt)
        num_text_tokens = await self.async_num_tokens_from_prompt(text)
        await asyncio.sleep(0.01)

        return num_prompt_tokens, num_text_tokens

    def count_tokens(self, input_prompt: str, text: str) -> Tuple[int, int]:
        if self.template == input_prompt:
            num_prompt_tokens = self.template_len
        else:
            num_prompt_tokens = self.num_tokens_from_prompt(input_prompt)
            self.template = input_prompt
            self.template_len = num_prompt_tokens

        num_text_tokens = self.num_tokens_from_prompt(text)

        return num_prompt_tokens, num_text_tokens
