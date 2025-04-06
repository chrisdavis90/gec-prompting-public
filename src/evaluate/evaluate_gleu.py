from omegaconf import OmegaConf
import os
import json
import glob
import logging
from multiprocessing import Pool
from tqdm import tqdm
import sys

from learner_datasets import get_dataset_info
from gleu import GLEU

logger = logging.getLogger(__name__)


def gleu_evaluate_jfleg_folder(folder: str, hyp_name: str):
    iter = 500
    n = 4
    sent = False

    # validate the folder exists
    # assume this folder is the specific run folder (e.g. run_1)
    if not os.path.exists(folder):
        failures.append(folder)
        raise Exception(f"Folder does not exist: {folder}")
    # validate the folder has a completion.txt file
    completion_file = os.path.join(folder, "completion.txt")
    if not os.path.exists(completion_file):
        failures.append(folder)
        raise Exception(f"completion.txt file does not exist in: {folder}")

    # load the config file
    exp_folder = os.path.dirname(folder)
    config_path = os.path.join(exp_folder, "config.yaml")
    # validate config file exists
    if not os.path.exists(config_path):
        failures.append(folder)
        raise Exception(f"config.yaml file does not exist in: {exp_folder}")

    args = OmegaConf.load(config_path)

    # get the dataset name
    dataset_name = args.dataset.name
    # get the dataset split
    dataset_split = args.dataset.split

    # get the dataset info
    dataset_info = get_dataset_info(dataset_name)

    # get the source file
    source_file = dataset_info.splits[dataset_split].file_path
    # get the reference file
    reference_files = dataset_info.splits[dataset_split].cor_file_path

    # get the hypothesis file
    hypothesis_file = os.path.join(folder, f"{hyp_name}.txt")

    # run gleu
    gleu_calculator = GLEU(n)
    gleu_calculator.load_sources(source_file)
    gleu_calculator.load_references(reference_files)
    gleu_output = [
        g
        for g in gleu_calculator.run_iterations(
            num_iterations=iter,
            source=source_file,
            hypothesis=hypothesis_file,
            per_sent=sent,
        )
    ][0]

    return gleu_output


def evaluate_folder(folder, hyp_name):
    # write the gleu output to a file
    gleu_output_file = os.path.join(
        folder,
        f"results_gleu_{hyp_name}.json",
    )

    logger.info(f"Processing {folder}")

    try:
        gleu_output = gleu_evaluate_jfleg_folder(folder, hyp_name)
    except Exception as e:
        print(e)
        return

    result = {
        "mean": gleu_output[0],
        "std": gleu_output[1],
        "ci1": gleu_output[2][0],
        "ci2": gleu_output[2][1],
    }

    with open(gleu_output_file, "w") as f:
        json.dump(result, f, indent=4)


def gleu_evaluation_main(
    base_dir, exp_folder, exp_pattern, hyp_name, p, run_pattern="run_1"
):
    folders = glob.glob(os.path.join(base_dir, exp_folder, exp_pattern, run_pattern))
    if len(folders) == 0:
        return

    pool_args = []
    for folder in folders:
        pool_args.append((folder, hyp_name))

    p = min(p, len(folders))
    with Pool(processes=p) as process_pool:
        process_pool.starmap(evaluate_folder, tqdm(pool_args, total=len(folders)))


if __name__ == "__main__":
    base_folder = "output_final"

    # get all the folders with "jfleg" in the name
    jfleg_folders = glob.glob(os.path.join(base_folder, "*jfleg*zero_shot*_6*"))
    hyp_name = "hyp_post"

    failures = []

    pool_args = zip(jfleg_folders, [hyp_name] * len(jfleg_folders))

    with Pool(processes=30) as p:
        p.map(evaluate_folder, tqdm(pool_args, total=len(jfleg_folders)))
