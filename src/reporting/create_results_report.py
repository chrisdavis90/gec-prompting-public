import yaml
import os
import json
import argparse
import pandas as pd
import glob
from tqdm import tqdm
from omegaconf import OmegaConf, ListConfig, DictConfig


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="output_prompt_filter")
    parser.add_argument(
        "--errant_file", "-e", type=str, default="results_hyp_post.json"
    )
    parser.add_argument("--gleu_file", "-g", type=str, default="results_gleu.json")
    parser.add_argument(
        "--m2scorer_file", "-m", type=str, default="results_hyp_post_m2scorer.json"
    )
    parser.add_argument("--output", "-o", type=str, default="results.csv")

    return parser


def print_json(file_path: str, title: str):
    if not os.path.exists(file_path):
        return {
            "TP": "---",
            "FP": "---",
            "FN": "---",
            "Prec": "---",
            "Rec": "---",
            "F0.5": "---",
        }
    with open(file_path) as f:
        data = json.load(f)

    return data


def main(path, errant_file, gleu_file, m2scorer_file, output):
    results_order = ["TP", "FP", "FN", "Prec", "Rec", "F0.5", "GLEU"]
    all_results = []
    # loop through folders in path
    print(path)
    for folder in tqdm(glob.glob(os.path.join(path, "*"))):
        if not os.path.isdir(folder):
            continue

        if not os.path.exists(os.path.join(folder, "config_detailed.yaml")):
            continue

        config = OmegaConf.load(os.path.join(folder, "config_detailed.yaml"))
        dataset_name = config.dataset.name
        dataset_name_str = dataset_name.split("_")[0]

        # for each sub folder
        for run_folder in glob.glob(os.path.join(folder, "run_1")):
            run_index = os.path.basename(run_folder).split("_")[-1]

            completion_file = os.path.join(run_folder, "completion.txt")
            if not os.path.exists(completion_file):
                continue

            # select conll results from m2scorer if they exist,
            # otherwise use results from errant
            # m2scorer_file = os.path.join(run_folder, m2scorer_file)

            if "conll" in folder and os.path.exists(
                os.path.join(run_folder, m2scorer_file)
            ):
                results_file = os.path.join(run_folder, m2scorer_file)
            else:
                results_file = os.path.join(run_folder, errant_file)

            results_name = os.path.basename(results_file).split(".")[0]

            results = print_json(results_file, f"{results_name}:")

            if config is None or results is None:
                continue

            results_list = [
                results_name,
                config.model,
                config.prompt_type,
                config.prompt_index,
                f"{dataset_name_str}-{config.dataset.split}",
                run_index,
                # config.params.temperature,
                # time_sec,
            ]

            # create a hash from dictionary
            # results_hash = hash(tuple(results.items()))

            # optionally add GLEU score
            gleu_score = "---"
            if "jfleg" in dataset_name:
                results_file = os.path.join(run_folder, gleu_file)
                gleu_results = print_json(results_file, f"gleu")
                if "mean" in gleu_results:
                    gleu_score = gleu_results["mean"]

            # convert results dict to list, ordered by key
            for key in results_order:
                if key == "GLEU":
                    results_list.append(gleu_score)
                else:
                    results_list.append(results[key])

            prompt_template = config.prompt_template
            if type(prompt_template) == ListConfig:
                template_str = []
                for item in prompt_template:
                    if type(item) == DictConfig:
                        for k, v in item.items():
                            template_str.append(k + ":")
                            template_str.append(v)

                prompt_template = " ".join(template_str)

            prompt_template = prompt_template.replace("\n", " ")
            results_list.append(prompt_template)

            all_results.append(results_list)

    # convert results to dataframe
    column_names = [
        "file",
        "model",
        "prompt_type",
        "prompt_index",
        "split",
        "run",
        # "temp",
        # "duration",
    ]
    column_names += results_order
    column_names.append("prompt_template")

    df = pd.DataFrame(all_results, columns=column_names)

    # sort dataframe by F0.5
    df = df.sort_values(by=["F0.5"], ascending=False)

    # save to csv
    df.to_csv(os.path.join(path, output), index=False)


if __name__ == "__main__":
    parser = create_args()

    args = parser.parse_args(
        # optional debug overrides
        # [
        #     "--path",
        #     "/path/to/projects/gec-prompting/paper_output/output_2_shot_coyne_dev",
        #     "--errant_file",
        #     "results_hyp_post_v2-3-3.json",
        #     "--gleu_file",
        #     "result_gleu_post.json",
        #     "--m2scorer_file",
        #     "results_hyp_post_m2scorer.json",
        #     "--output",
        #     "results.csv",
        # ]
    )

    main(
        args.path,
        args.errant_file,
        args.gleu_file,
        args.m2scorer_file,
        args.output,
    )
