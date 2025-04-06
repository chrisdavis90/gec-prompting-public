import yaml
import os
import json
import argparse
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# Connect the path with your '.env' file name
load_dotenv(os.path.join(BASEDIR, ".env"))

from learner_datasets import get_dataset_info

"""
A file to produce a log report from a run.
The report should include:
- model name
- dataset name and split
- prompt type
- prompt template
- ERRANT evaluation
- original, hypothesis, reference sentences
"""


def read_m2_sentences(file_path, n=20):
    sentences = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if line.startswith("S "):
                sentences.append(line[2:].strip())
            if len(sentences) >= n:
                break

    return sentences


def read_sentences(file_path, n=20):
    sentences = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            sentences.append(line.strip())
            if len(sentences) >= n:
                break

    return sentences


def create_args():
    parser = argparse.ArgumentParser()

    # todo: add validation

    # data
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="model_output_v2/wibea_dataset_train20_command_zero_shot_9_temp=0.1_topk=50_topp=1.0/run_1",
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("-r", "--results", type=str, default="results_hyp.json")

    return parser


def print_json(file_path: str, title: str):
    if not os.path.exists(file_path):
        return
    with open(file_path) as f:
        data = json.load(f)
    print(title)
    row = []
    for k in ["TP", "FP", "FN", "Prec", "Rec", "F0.5"]:
        row.append(data[k])
    print(" ".join([str(x) for x in row]))
    # print(json.dumps(data, indent=4))
    print("")
    return data


def main(path: str, n: int, results_name: str = "results.json"):
    base_path = os.path.dirname(path)

    config_path = os.path.join(base_path, "config_detailed.yaml")
    config = OmegaConf.load(config_path)

    if config is None:
        print(f"No config file found at {config_path}")
        exit()

    # 1. Print Config
    print(json.dumps(OmegaConf.to_container(config, resolve=True), indent=4))
    # print(OmegaConf.to_yaml(config, resolve=True))

    # 2. Print Results
    results_file = os.path.join(path, results_name)
    if not os.path.exists(results_file):
        print(f"No results file found at {results_file}")
        exit()

    _ = print_json(results_file, "Results:")

    # optional all results file
    results_name = os.path.basename(results_file).split(".")[0]
    all_results_file = os.path.join(path, f"all_{results_name}.json")
    # json load all results
    sentence_results = None
    if os.path.exists(all_results_file):
        with open(all_results_file) as f:
            all_results = json.load(f)
        sentence_results = all_results["sentence_results"]

    # print sentences...
    dataset_name = config.dataset.name
    dataset_split = config.dataset.split
    dataset_info = get_dataset_info(dataset_name)

    original_file = dataset_info.splits[dataset_split].file_path
    original_sentences = read_sentences(original_file, n)

    results_name = os.path.basename(results_file).split(".")[0]
    hyp_name = "hyp"
    if "_" in results_name:
        hyp_name = results_name.split("_")
        hyp_name = "_".join(hyp_name[1:])

    hypothesis_file = os.path.join(path, f"{hyp_name}.txt")
    hypothesis_sentences = read_sentences(hypothesis_file, n)

    reference_file = dataset_info.splits[dataset_split].cor_file_path
    reference_sentences = read_sentences(reference_file, n)

    output_file = os.path.join(path, f"output_{hyp_name}.txt")
    with open(output_file, "wt") as outf:
        if sentence_results is not None:
            sentence_results = sentence_results[:n]
            assert len(sentence_results) == len(original_sentences)
            for i, (o, h, r) in enumerate(
                zip(original_sentences, hypothesis_sentences, reference_sentences)
            ):
                lines = [
                    f"Original: {o}",
                    f"Hypothesis: {h}",
                    f"Reference: {r}",
                    f'Hypothesis Edits: {sentence_results[i]["HYPOTHESIS EDITS"]}',
                    f'Reference Edits: {sentence_results[i]["REFERENCE EDITS"]}',
                    f'Local TP/FP/FN: {sentence_results[i]["Local TP/FP/FN"]}',
                    f'Local P/R/F0.5: {sentence_results[i]["Local P/R/F0.5"]}',
                    f'Global TP/FP/FN: {sentence_results[i]["Global TP/FP/FN"]}',
                    f'Global P/R/F0.5: {sentence_results[i]["Global P/R/F0.5"]}',
                ]
                outf.write("\n".join(lines))
                outf.write("\n\n")

                print("\n".join(lines))
                print("\n")
        else:
            for o, h, r in zip(
                original_sentences, hypothesis_sentences, reference_sentences
            ):
                lines = [
                    f"Original: {o}",
                    f"Hypothesis: {h}",
                    f"Reference: {r}",
                ]
                outf.write("\n".join(lines))
                outf.write("\n")

                print("\n".join(lines))


if __name__ == "__main__":
    parser = create_args()
    args = parser.parse_args()
    main(args.path, args.n, args.results)
