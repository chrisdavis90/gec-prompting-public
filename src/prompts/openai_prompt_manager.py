import os
from .prompt_manager import PromptManager

"""define a class to retrieve text from prompt.json"""


class OpenAIPromptManager(PromptManager):
    # initialise
    def __init__(self, model_name=None, prompt_path=None, len_multiplier=1.5):
        # initialise parent class
        prompt_path = os.path.join(os.path.dirname(__file__), "openai_prompts.json")
        super().__init__(prompt_path=prompt_path, len_multiplier=len_multiplier)

    def _format_user_message(self, user_message: str, text: str) -> str:
        if user_message is None and text is None:
            return ""
        elif user_message is None:
            # text is not None
            return text
        elif text is None:
            # user_message is not None
            return user_message

        # both are not None
        if "{text}" in user_message:
            return user_message.format(text=text)
        else:
            return f"{user_message.strip()} {text}"

    def _format_prompt_message(self, message: dict, role: str) -> dict:
        return {"role": role, "content": message.get(role)}

    def format_prompt(self, prompt_template: list, input_text: str = None):
        messages = []

        added_input_text = False
        for i, message in enumerate(prompt_template):
            if "system" in message:
                messages.append(self._format_prompt_message(message, "system"))
            elif "assistant" in message:
                messages.append(self._format_prompt_message(message, "assistant"))
            elif "user" in message:
                if i == len(prompt_template) - 1:
                    # last message
                    user_message = self._format_user_message(
                        message["user"], input_text
                    )
                    messages.append(
                        self._format_prompt_message({"user": user_message}, "user")
                    )
                    added_input_text = True
                else:
                    messages.append(self._format_prompt_message(message, "user"))

        if not added_input_text and input_text is not None:
            user_message = self._format_user_message(None, input_text)

            messages.append(self._format_prompt_message({"user": user_message}, "user"))

        return messages

    # generator to iterate over prompts
    def iterate_prompts(self, prompt_type: str = None):
        if prompt_type is None:
            for prompt_type in self.prompt_dict.keys():
                yield from self.iterate_prompts(prompt_type)
        else:
            prompt_list = self.prompt_dict[prompt_type]
            for prompt in prompt_list:
                yield prompt
