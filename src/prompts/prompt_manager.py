from typing import Union
import os
import json
import random

"""define a class to retrieve text from a *prompt.json"""


class PromptManager:
    _model_to_prompt_file = {
        "meta-llama/Llama-2-7b-chat-hf": "llama_chat_prompts.json",
        "meta-llama/Llama-2-13b-chat-hf": "llama_chat_prompts.json",
        "meta-llama/Llama-2-70b-chat-hf": "llama_chat_prompts.json",
        "stabilityai/StableBeluga2": "stable_beluga_prompts.json",
        "Writer/InstructPalmyra-20b": "instruct_palmyra_prompts.json",
    }

    DEFAULT_PROMPT_FILE = "default_prompts.json"

    # initialise
    def __init__(self, model_name=None, prompt_path=None, len_multiplier=1.5):
        self.len_multiplier: float = len_multiplier

        prompt_file_name = self._model_to_prompt_file.get(
            model_name,
            self.DEFAULT_PROMPT_FILE,
        )

        if prompt_path is None:
            self.prompt_path = os.path.join(
                os.path.dirname(__file__),
                prompt_file_name,
            )
        else:
            self.prompt_path = prompt_path

        self.prompt_dict = self.load_prompts()

    # load prompts
    def load_prompts(self) -> dict:
        with open(self.prompt_path, "r") as f:
            prompt_dict = json.load(f)
        return prompt_dict

    # get prompt types
    def get_prompt_types(self) -> list:
        return list(self.prompt_dict.keys())

    # get prompt
    def get_prompt(self, prompt_type, index) -> Union[str, list]:
        return self.prompt_dict[prompt_type][index]

    # sample prompt
    def sample_prompt(self, prompt_type) -> Union[str, list]:
        prompt_list = self.prompt_dict[prompt_type]
        return random.choice(prompt_list)

    def format_prompt(self, prompt_template, input_text) -> str:
        # check if {text} in template
        if "{text}" not in prompt_template:
            return prompt_template.strip() + " " + input_text.strip()

        # replace {text} with input_text
        return prompt_template.format(text=input_text).strip()

    def format_template(self, prompt_template, few_shot_examples) -> str:
        
        for i, example_pair in enumerate(few_shot_examples):
            sentence, corrected = example_pair
            prompt_template = prompt_template.replace(f"{{sentence_{i+1}}}", sentence)
            prompt_template = prompt_template.replace(f"{{corrected_{i+1}}}", corrected)

        return prompt_template

    # generator to iterate over prompts
    def iterate_prompts(self, prompt_type: str = None):
        if prompt_type is None:
            for prompt_type in self.prompt_dict.keys():
                yield from self.iterate_prompts(prompt_type)
        else:
            prompt_list = self.prompt_dict[prompt_type]
            for prompt in prompt_list:
                yield prompt
