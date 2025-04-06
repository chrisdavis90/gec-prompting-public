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
from models.base import BaseModel
from models.response import ResponseObject
from models.status_tracker import StatusTracker

logger = logging.getLogger(__name__)


class APIModel(BaseModel):
    # initialize
    def __init__(self, model_name, output_dir=None):
        # call super constructor
        super().__init__(model_name, output_dir)

        self.client = None
        self.async_client = None

        self.log_file = None
        self.api_request_cls = None
        # constants to move to init
        self.seconds_to_pause_after_rate_limit_error = 15
        self.seconds_to_sleep_each_loop = 0.05
        # ^ 1 ms limits max throughput to 100 requests per second

        self.max_requests_per_minute = 500
        self.max_tokens_per_minute = 70000
        self.max_attempts = 15

    async def _run_batch(self, prompt_info_batch: list) -> list:
        # initialize trackers
        await asyncio.sleep(0.01)
        queue_of_requests_to_retry = asyncio.Queue()
        status_tracker = (
            StatusTracker()
        )  # single instance to track a collection of variables

        requests = []  # variable to hold results of all requests
        api_requests = []
        next_request = None  # variable to hold the next request to call

        # initialize available capacity counts
        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()

        # initialize flags
        batch_not_finished = True  # after file is empty, we'll skip reading it

        logger.debug("Initialization complete.")

        # create iterator from list of prompts
        prompt_info_batch = iter(prompt_info_batch)

        while True:
            # logger.debug("Main loop started.")
            # get next request if one is not already waiting for capacity
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logger.debug(f"Retrying task {next_request.sentence_id}.")
            elif batch_not_finished:
                try:
                    # get new request
                    prompt_info = next(prompt_info_batch)
                    n_tokens_from_prompt = self.num_tokens_from_prompt(
                        prompt_info["prompt"]
                    )
                    next_request = self.api_request_cls(
                        sentence_id=prompt_info["sentence_id"],
                        token_consumption=(
                            n_tokens_from_prompt
                            + prompt_info["kwargs"].get("max_tokens", 128)
                        ),
                        attempts_left=self.max_attempts,
                        prompt=prompt_info["prompt"],
                        kwargs=prompt_info["kwargs"],
                        client=self.async_client,
                        model_name=self.model_name,
                    )

                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logger.debug(f"Starting task {next_request.sentence_id}.")
                except StopIteration:
                    # no more requests to make
                    batch_not_finished = False
                    next_request = None
                    logger.debug("No more requests to make.")

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity += min(
                available_request_capacity
                + self.max_requests_per_minute * seconds_since_update / 60.0,
                self.max_requests_per_minute,
            )
            available_token_capacity += min(
                available_token_capacity
                + self.max_tokens_per_minute * seconds_since_update / 60.0,
                self.max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API and keep handle to task
                    requests.append(
                        asyncio.create_task(
                            next_request.call_api(
                                retry_queue=queue_of_requests_to_retry,
                                status_tracker=status_tracker,
                            )
                        )
                    )
                    # asyncio.create_task(next_request.call_api())
                    # api_requests = next_request
                    next_request = None

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps so concurrent tasks can run
            await asyncio.sleep(self.seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (
                time.time() - status_tracker.time_of_last_rate_limit_error
            )
            if (
                seconds_since_rate_limit_error
                < self.seconds_to_pause_after_rate_limit_error
            ):
                remaining_seconds_to_pause = (
                    self.seconds_to_pause_after_rate_limit_error
                    - seconds_since_rate_limit_error
                )
                logging.warn(
                    f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + self.seconds_to_pause_after_rate_limit_error)}"
                )
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago

        # process request_results
        results = {}
        for request in requests:
            req_result = request.result()
            req_exception = request.exception()
            if req_exception:
                a = 1
            if req_result and not req_exception:
                response = request.result()

                results[response.sentence_id] = response

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
        num_prompt_tokens = self.num_tokens_from_messages(input_prompt)

        # num_text_tokens reflects the number of tokens for the
        #   user input text only, excluding any constants added by
        #   OpenAI ChatML. This excludes self.tokens_per_message and
        #   self.tokens_per_reply
        num_text_tokens = len(self._encoding.encode(text))

        return num_prompt_tokens, num_text_tokens
