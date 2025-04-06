from typing import List, Tuple, Union
import time


class BaseModel:
    def __init__(self, model_name, output_dir=None):
        self.model_name = model_name
        self.output_dir = output_dir

    def run(self, prompt, **kwargs) -> str:
        raise NotImplementedError()

    def count_tokens(
        self,
        input_prompt: Union[List, str],
        text: str,
    ) -> Tuple[int, int]:
        raise NotImplementedError()

    def sleep(self):
        time.sleep(self.sleep_time)
