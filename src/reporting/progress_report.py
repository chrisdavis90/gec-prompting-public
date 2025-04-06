import os
import glob
import json
import pandas as pd


def progress_zero_shot(errant_result_file, gleu_result_file):
    output_folder = "output_final"

    models = [
        "facebook/opt-iml-max-30b",
        "stabilityai/StableBeluga2",  # 70B, uses a special prompt format
        "tiiuae/falcon-40b-instruct",
        "google/flan-t5-xxl",  # 11.3b
        "bigscience/bloomz-7b1",
        "Writer/InstructPalmyra-20b",
        "gpt-3.5-turbo-0613",
        "command",
        "meta-llama/Llama-2-70b-chat-hf",
        "gpt-4-0613",
        # "databricks/dolly-v2-12b",
    ]
    datasets = {
        "wibea": ["dev"],
        "fce": ["dev", "test"],
        "jfleg": ["dev", "test"],
        "conll14": ["test"],
    }

    prompt_types = {
        "zero-shot": [2, 4, 5, 6, 7, 9, 10],
        "zero-shot-instruct-input": [2, 4, 5, 6, 7, 9, 10],
        "few-shot": [0, 3],
    }

    all_results = []

    column_names = ["model", "prompt_type", "prompt_index"]
    for dataset_name in sorted(datasets.keys()):
        for split in sorted(datasets[dataset_name]):
            dname = f"{dataset_name}-{split}"
            column_names.append(dname)
    # for zero-shot

    for model in models:
        model_name = model.split("/")[-1]
        for prompt_index in prompt_types["zero-shot"]:
            row = [model]

            prompt_name = "zero_shot"
            if "InstructPalmyra" in model:
                prompt_name = "zero_shot_instruct_input"

            row.append(prompt_name)
            row.append(prompt_index)

            for dataset_name in sorted(datasets.keys()):
                for split in sorted(datasets[dataset_name]):
                    folder_pattern = f"{dataset_name}_dataset_{split}_{model_name}_{prompt_name}_{prompt_index}*"

                    folders = glob.glob(os.path.join(output_folder, folder_pattern))

                    if len(folders) == 0:
                        row.append("NA")
                    else:
                        assert len(folders) == 1
                        folder = folders[0]

                        if dataset_name == "jfleg":
                            results_file = os.path.join(
                                folder, "run_1", gleu_result_file
                            )
                        else:
                            results_file = os.path.join(
                                folder, "run_1", errant_result_file
                            )

                        if not os.path.exists(results_file):
                            row.append("NA")
                        else:
                            with open(results_file) as f:
                                results = json.load(f)

                            if dataset_name == "jfleg":
                                row.append(results["mean"])
                            else:
                                row.append(results["F0.5"])

            all_results.append(row)

    df = pd.DataFrame(all_results, columns=column_names)

    # sort dataframe by model, then by prompt type, then by prompt index
    df = df.sort_values(by=["model", "prompt_type", "prompt_index"])

    df.to_csv("progress_zeroshot.csv", index=False)


def progress_few_shot(errant_result_file, gleu_result_file):
    output_folder = "output_final"

    models = {
        "facebook/opt-iml-max-30b": {
            "few-shot": [0, 1, 2, 3],
        },
        "stabilityai/StableBeluga2": {
            "few-shot": [0, 1, 2, 3, 4],
        },
        "tiiuae/falcon-40b-instruct": {
            "few-shot": [0, 1, 2, 3],
        },
        "google/flan-t5-xxl": {
            "few-shot": [0, 1, 2, 3],
        },
        "bigscience/bloomz-7b1": {
            "few-shot": [0, 1, 2, 3],
        },
        "Writer/InstructPalmyra-20b": {
            "few-shot-instruct-input": [0, 1, 2, 3, 4],
        },
        "gpt-3.5-turbo-0613": {
            "few-shot-chat": [0, 1, 2, 3, 4, 5],
        },
        "gpt-4-0613": {
            "few-shot-chat": [0, 1, 2, 3, 4, 5],
        },
        "command": {
            "few-shot": [0, 1, 2, 3],
        },
        # "databricks/dolly-v2-12b": {
        #     "few-shot": [0, 1, 2, 3],
        # },
        "meta-llama/Llama-2-70b-chat-hf": {
            "few-shot-in": [0, 1, 2, 3],
        },
    }

    datasets = {
        "wibea": ["dev"],
        "fce": ["dev", "test"],
        "jfleg": ["dev", "test"],
        "conll14": ["test"],
    }

    # prompt_types = {
    #     "few-shot-instruct-input": [0, 1, 2, 3, 4],
    #     "few-shot": [0, 1, 2, 3, 4],
    # }

    all_results = []

    column_names = ["model", "prompt_type", "prompt_index"]
    for dataset_name in sorted(datasets.keys()):
        for split in sorted(datasets[dataset_name]):
            dname = f"{dataset_name}-{split}"
            column_names.append(dname)
    # for zero-shot

    for model, prompt_info in models.items():
        model_name = model.split("/")[-1]

        prompt_name = list(prompt_info.keys())[0]
        prompt_indices = prompt_info[prompt_name]

        for prompt_index in prompt_indices:
            row = [model]

            prompt_name_str = prompt_name.replace("-", "_")

            row.append(prompt_name_str)
            row.append(prompt_index)

            for dataset_name in sorted(datasets.keys()):
                for split in sorted(datasets[dataset_name]):
                    folder_pattern = f"{dataset_name}_dataset_{split}_{model_name}_{prompt_name_str}_{prompt_index}*"

                    folders = glob.glob(os.path.join(output_folder, folder_pattern))

                    if len(folders) == 0:
                        row.append("NA")
                    else:
                        assert len(folders) == 1
                        folder = folders[0]

                        if dataset_name == "jfleg":
                            results_file = os.path.join(
                                folder, "run_1", gleu_result_file
                            )
                        else:
                            results_file = os.path.join(
                                folder, "run_1", errant_result_file
                            )

                        if not os.path.exists(results_file):
                            row.append("NA")
                        else:
                            with open(results_file) as f:
                                results = json.load(f)

                            if dataset_name == "jfleg":
                                row.append(results["mean"])
                            else:
                                row.append(results["F0.5"])

            all_results.append(row)

    df = pd.DataFrame(all_results, columns=column_names)

    # sort dataframe by model, then by prompt type, then by prompt index
    df = df.sort_values(by=["model", "prompt_type", "prompt_index"])

    df.to_csv("progress_fewshot.csv", index=False)
