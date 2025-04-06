import asyncio
import logging
import os
import time

from typing import List, Tuple

import jsonlines
import openai
import tiktoken
from openai.error import RateLimitError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from models.api_request import APIRequest
from models.api_model import APIModel
from models.response import ResponseObject
from models.status_tracker import StatusTracker

logger = logging.getLogger(__name__)


class OpenAIRequest(APIRequest):
    async def _call_api(
        self,
        status_tracker: StatusTracker,
    ):
        await asyncio.sleep(0.01)
        error = None
        completion = None
        try:
            # call api
            logger.debug(f"Calling API for sentence {self.sentence_id}.")
            async with asyncio.timeout(30):
                completion = await openai.ChatCompletion.acreate(
                    model=self.model_name,
                    messages=self.prompt,
                    **self.kwargs,
                )
            logger.debug(f"API call complete for sentence {self.sentence_id}.")
        except TimeoutError as e:
            error = f"Timeout error on sentence {self.sentence_id}: {e}"
            logger.warning(error)
            status_tracker.num_api_errors += 1
        except RateLimitError as e:
            status_tracker.num_rate_limit_errors += 1
            status_tracker.time_of_last_rate_limit_error = time.time()
            error = f"Error on sentence {self.sentence_id}: {e}"
            logger.warning(error)
            status_tracker.num_api_errors += 1
        except Exception as e:
            error = f"Error on sentence {self.sentence_id}: {e}"
            logger.warning(error)
            status_tracker.num_api_errors += 1

        return (error, completion)


class OpenAIModel(APIModel):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # initialize
    def __init__(self, model_name, output_dir=None):
        # call super constructor
        super().__init__(model_name, output_dir)

        self.log_file = None
        if output_dir is not None:
            self.log_file = os.path.join(output_dir, "usage.jsonl")
            self.response_info_file = os.path.join(
                output_dir,
                "response_info.jsonl",
            )
        self.sleep_time = 1
        self._encoding = tiktoken.encoding_for_model(self.model_name)
        self._set_token_constants(model_name=self.model_name)

        self.api_request_cls = OpenAIRequest

        # constants to move to init
        self.seconds_to_pause_after_rate_limit_error = 30
        self.seconds_to_sleep_each_loop = 0.1
        # ^ 1 ms limits max throughput to 10 requests per second

        if "gpt-3.5-turbo" in self.model_name:
            self.max_requests_per_minute = 500
            self.max_tokens_per_minute = 70000
        elif "gpt-4" in self.model_name:
            self.max_requests_per_minute = 180
            self.max_tokens_per_minute = 8000

        logger.info(f"Initialized OpenAIModel with model {self.model_name}.")
        logger.info(f"Max requests per minute: {self.max_requests_per_minute}.")
        logger.info(f"Max tokens per minute: {self.max_tokens_per_minute}.")

        self.max_attempts = 15

    def _set_token_constants(self, model_name=None) -> None:
        model = self.model_name if model_name is None else model_name

        if "gpt-3.5-turbo" in model:
            print(
                "Warning: gpt-3.5-turbo may update over time. \
                  Setting num tokens assuming gpt-3.5-turbo-0613."
            )
            model = "gpt-3.5-turbo-0613"
        elif "gpt-4" in model:
            print(
                "Warning: gpt-4 may update over time. \
                  Setting num tokens assuming gpt-4-0613."
            )
            model = "gpt-4-0613"

        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for \
                    model {model}. See https://github.com/openai/openai\
                    -python/blob/main/chatml.md for information on how \
                    messages are converted to tokens."""
            )

        self.tokens_per_message = tokens_per_message
        self.tokens_per_name = tokens_per_name

        #### NOTE: there are contradictions in the documentation of whether
        ####   to set tokens_per_reply to 2 or 3.
        #### In testing, prompt-template estimates were off by one when
        ####   using tokens_per_reply=2

        # according to Microsoft article and OpenAI (managing-tokens source)
        # every reply is primed with <im_start>assistant
        # self.tokens_per_reply = 2

        # according to OpenAI (How to format inputs)
        # every reply is primed with <|start|>assistant<|message|>
        self.tokens_per_reply = 3

    def num_tokens_from_prompt(self, messages: List) -> int:
        """Return the number of tokens used by a list of messages.
        This is an approximation because there are inconsistencies between
            sources, and the underlying implementation may change.
        OpenAI source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        Microsoft source: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions
        Another OpenAI source: https://platform.openai.com/docs/guides/gpt/managing-tokens
        """

        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                num_tokens += len(self._encoding.encode(value))
                if key == "name":
                    num_tokens += self.tokens_per_name
        num_tokens += self.tokens_per_reply
        return num_tokens

    def log_response_info(self, completion) -> None:
        if self.log_file is None:
            return

        with jsonlines.open(self.log_file, "a") as f:
            f.write(completion["usage"])

        response_object = completion.copy()
        response_object.pop("usage", None)
        choices = response_object.pop("choices", None)
        if choices:
            finish_reason = choices[0].pop("finish_reason", None)
            if finish_reason:
                response_object["finish_reason"] = finish_reason

        with jsonlines.open(self.response_info_file, "a") as f:
            f.write(response_object)

    # run
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_random_exponential(
            min=1,
            max=60,
        ),
    )
    def run_single(self, prompt, **kwargs) -> str:
        try:
            completion = openai.ChatCompletion.create(
                model=self.model_name, messages=prompt, **kwargs
            )
            self.log_response_info(completion)
        except Exception as e:
            raise e

        return completion["choices"][0]["message"]["content"]

    def run_batch(self, prompt_info_batch: list) -> None:
        results = asyncio.run(self._run_batch(prompt_info_batch))

        # log info and update output format
        for sentence_id, response in results.items():
            if response.output:
                self.log_response_info(response.output)

                response.output = response.output["choices"][0]["message"]["content"]

        return results

    def count_tokens(self, input_prompt: List, text: str) -> Tuple[int, int]:
        """Returns the number of tokens in the given text, encoded using tiktoken
        See https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
            for example code using tiktoken.
        See https://github.com/openai/openai-python/blob/main/chatml.md
            for information on how messages are converted to tokens.
        """
        # num_prompt_tokens reflects the number of tokens for the prompt
        #   template. As such, this includes self.tokens_per_reply --
        #   a constant number of tokens for the model reply.
        num_prompt_tokens = self.num_tokens_from_prompt(input_prompt)

        # num_text_tokens reflects the number of tokens for the
        #   user input text only, excluding any constants added by
        #   OpenAI ChatML. This excludes self.tokens_per_message and
        #   self.tokens_per_reply
        num_text_tokens = len(self._encoding.encode(text))

        return num_prompt_tokens, num_text_tokens
