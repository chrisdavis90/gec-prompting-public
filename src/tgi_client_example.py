from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from huggingface_hub import InferenceClient

from learner_datasets import get_dataset_info

if __name__ == "__main__":
    """
    This script demonstrates how to use the InferenceClient to interact with
    the Text Generation Inference server.
    Ensure the server has started before running this script.
    See src/scripts/launch_tgi_server.sh for instructions on how to
     start the server.
    """

    # Initialise TGI client
    client = InferenceClient(model="http://0.0.0.0:8080")

    # Load dataset
    dataset_name = "wifce_dataset"
    eval_name = "train2000"
    dataset_info = get_dataset_info(dataset_name)
    eval_dataset = load_dataset(
        f"src/learner_datasets/{dataset_name}",
        split=eval_name,
    )

    # Set prompt template
    prompt_template = "Reply with a corrected version of the input sentence \
        with all grammatical and spelling errors fixed. If there are no \
            errors, reply with a copy of the original sentence.\n\nInput \
                sentence: {text}\nCorrected sentence: "
    
    print("Prompt template:")
    print(prompt_template)

    # Iterate over dataset and generate responses
    generations = []
    for text in KeyDataset(eval_dataset, "sentence"):
        prompt = prompt_template.format(text=text)

        output = client.text_generation(prompt="What is Deep Learning?")

        # With arguments:
        # output = client.text_generation(
        #     prompt=prompt,
        #     return_full_text=False,
        #     temperature=1,
        #     do_sample=True,
        #     top_k=50,
        #     seed=42,
        #     # need to increase max tokens
        # )

        generations.append(output)

        print(f"Input: {text}")
        print(f"Output: {output}")
