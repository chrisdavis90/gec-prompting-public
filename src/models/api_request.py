import time
import asyncio
import logging
from typing import Union
from dataclasses import dataclass, field

from models.status_tracker import StatusTracker
from models.response import ResponseObject
from openai.error import RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class APIRequest:
    model_name: str
    sentence_id: int
    token_consumption: int
    attempts_left: int
    prompt: Union[list, str]
    kwargs: dict
    result: dict = None
    error: list = field(default_factory=list)
    client: object = None

    async def _call_api(self, status_tracker: StatusTracker):
        raise NotImplementedError

    async def call_api(
        self,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
    ) -> ResponseObject:
        t0 = time.time()

        error, completion = await asyncio.create_task(
            self._call_api(status_tracker),
        )

        if error:
            self.result = None
            self.error.append(error)
            if self.attempts_left:
                # add to retry queue
                retry_queue.put_nowait(self)
            else:
                logger.error(f"Task {self.sentence_id} failed after all attempts.")
                status_tracker.num_tasks_failed += 1
                status_tracker.num_tasks_in_progress -= 1
                self.error.append(f"Task {self.sentence_id} failed after all attempts.")

                # raise Exception(f"Task {self.sentence_id} failed.")
        else:
            # success
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            self.result = completion
            logger.debug(f"Sentence {self.sentence_id} succeeded.")

        t1 = time.time()
        duration = t1 - t0
        logger.debug(f"Task {self.sentence_id} took {duration} seconds.")

        return ResponseObject(
            sentence_id=self.sentence_id,
            output=self.result,
            errors=self.error,
            duration=duration,
        )
