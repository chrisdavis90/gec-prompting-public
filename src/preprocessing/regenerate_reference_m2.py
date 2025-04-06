import glob
import json
import logging
import os
import subprocess
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm


# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(BASEDIR, ".env"))

# try and import the learner_datasets module
try:
    from learner_datasets import get_dataset_info, DatasetEnum
except ImportError:
    # if the module is not found, add the src directory to the path
    import sys

    sys.path.append(os.path.join(BASEDIR, "src"))
    from learner_datasets import get_dataset_info

logger = logging.getLogger(__name__)


def regenerate_default():
    for dataset in DatasetEnum:
        dataset_info = get_dataset_info(dataset)

        folder = os.path.join(dataset_info.CPATH, "m2")
        new_ref_dir = os.path.join(folder, "m2_v2-3-3")
        if not os.path.exists(new_ref_dir):
            os.makedirs(new_ref_dir)

        for split_name, split_info in dataset_info.splits.items():
            original_file = split_info.file_path
            corrected_file = split_info.cor_file_path
            m2_file = split_info.ref_file_path

            m2_file_name = os.path.basename(m2_file)
            new_m2_file = os.path.join(new_ref_dir, m2_file_name)

            # call errant parallel
            errant_parallel_args = [
                "errant_parallel",
                "-orig",
                original_file,
                "-out",
                new_m2_file,
                "-tok",
                "-lev",
            ]

            errant_parallel_args.append("-cor")
            if type(corrected_file) is str:
                errant_parallel_args.append(corrected_file)
            elif type(corrected_file) is list:
                errant_parallel_args.extend(corrected_file)

            subprocess.run(errant_parallel_args)
            print(new_m2_file)


def regenerate_cefr_refs():
    dataset = DatasetEnum.wibea_dataset
    dataset_info = get_dataset_info(dataset)

    folder = os.path.join(dataset_info.CPATH, dataset_info.folder)
    new_ref_dir = os.path.join(folder, "m2_v2-3-3")
    if not os.path.exists(new_ref_dir):
        os.makedirs(new_ref_dir)

    cefr_files = ["A.dev", "B.dev", "C.dev", "N.dev"]
    split_info = dataset_info.splits["dev"]

    for cefr_file in cefr_files:
        original_file = os.path.join(dataset_info.DATA_DIR, f"{cefr_file}.detok.orig")
        corrected_file = os.path.join(dataset_info.COR_DIR, f"{cefr_file}.detok.cor")

        m2_file = f"{cefr_file}.auto.m2"
        new_m2_file = os.path.join(new_ref_dir, m2_file)

        # call errant parallel
        errant_parallel_args = [
            "errant_parallel",
            "-orig",
            original_file,
            "-out",
            new_m2_file,
            "-tok",
            "-lev",
        ]

        errant_parallel_args.append("-cor")
        if type(corrected_file) is str:
            errant_parallel_args.append(corrected_file)
        elif type(corrected_file) is list:
            errant_parallel_args.extend(corrected_file)

        subprocess.run(errant_parallel_args)
        print(new_m2_file)


if __name__ == "__main__":
    regenerate_cefr_refs()
