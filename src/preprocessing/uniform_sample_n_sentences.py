import os
import numpy as np
import random
from learner_datasets import WIBEAInfo, FCEInfo


def sample_sentences_from_m2(file_path: str, indices: list) -> list:
    # read in the data from a file
    with open(file_path, "r") as f:
        data = f.read()
        # split the data into sentences by empty lines
        sentences = np.array(data.split("\n\n"))
        sentences = sentences[indices]

    return list(sentences)


def sample_sentences(file_path: str, indices: list) -> list:
    sentences = []
    with open(file_path, "r") as f:
        data = f.readlines()
        sentences = np.array(data)[indices]

    return list(sentences)


def process_file(file_path: str, indices: list, out_file_path: str):
    sentences = sample_sentences(file_path, indices)

    with open(out_file_path, "a") as f:
        f.writelines(sentences)

    return sentences


def process_m2_file(file_path: str, indices: list, out_file_path: str):
    sentences = sample_sentences_from_m2(file_path, indices)

    with open(out_file_path, "a") as f:
        for sentence in sentences:
            f.write(sentence + "\n\n")

    return sentences


def load_indices_from_file(file_path: str) -> list:
    return np.load(file_path)


def sample_indices(num_lines: int, n_sents: int) -> list:
    return random.sample(range(num_lines), n_sents)


def sample_from_dataset(dataset_info, n_sents: int, out_dir: str, name: str):
    n_sents = 1000

    split = "train"
    split_info = dataset_info.splits[split]

    # "ABC.train.detok.orig"
    orig_file_path = split_info.file_path
    cor_file_path = split_info.cor_file_path
    ref_file_path = split_info.ref_file_path  # m2 file

    # count lines in file
    with open(orig_file_path, "r") as f:
        num_lines = sum(1 for _ in f)

    # sample n_sents lines
    sampled_indices = random.sample(range(num_lines), n_sents)

    # save indices in root dir
    # base_dir = os.path.dirname(orig_file_path)
    indices_file = os.path.join(
        out_dir, f"{dataset_info.folder}.train.n{n_sents}.indices.npy"
    )
    # sampled_indices = load_indices_from_file(indices_file)
    np.save(
        indices_file,
        sampled_indices,
    )

    origsents = process_file(
        orig_file_path,
        sampled_indices,
        os.path.join(out_dir, "orig", f"{name}.train.detok.n{n_sents}.orig"),
    )

    corsents = process_file(
        cor_file_path,
        sampled_indices,
        os.path.join(out_dir, "cor", f"{name}.train.detok.n{n_sents}.cor"),
    )

    m2sents = process_m2_file(
        ref_file_path,
        sampled_indices,
        os.path.join(out_dir, "m2", f"{name}.train.auto.n{n_sents}.m2"),
    )

    return origsents, corsents, m2sents


if __name__ == "__main__":
    """
    Uniformly sample n_sents from W&I and FCE and save them as a new dataset
    """
    CPATH = os.getenv("CORPORA")
    out_dir = os.path.join(CPATH, "wifce")

    # if out_dir already exists, then exit
    if os.path.exists(out_dir):
        print(f"{out_dir} already exists. Exiting.")
        exit()

    wandiorig, wandicor, wandm2 = sample_from_dataset(
        dataset_info=WIBEAInfo,
        n_sents=1000,
        out_dir=out_dir,
        name="wandi",
    )

    fceorig, fcecor, fcem2 = sample_from_dataset(
        dataset_info=FCEInfo,
        n_sents=1000,
        out_dir=out_dir,
        name="fce",
    )

    # merge the two sets of lists
    origsents = wandiorig + fceorig
    corsents = wandicor + fcecor
    m2sents = wandm2 + fcem2

    # save the merged lists
    out_file_path = os.path.join(out_dir, "orig", "train.detok.n2000.orig")
    with open(out_file_path, "w") as f:
        f.writelines(origsents)

    out_file_path = os.path.join(out_dir, "cor", "train.detok.n2000.cor")
    with open(out_file_path, "w") as f:
        f.writelines(corsents)

    out_file_path = os.path.join(out_dir, "m2", "train.auto.n2000.m2")
    with open(out_file_path, "w") as f:
        for sentence in m2sents:
            f.write(sentence + "\n\n")
