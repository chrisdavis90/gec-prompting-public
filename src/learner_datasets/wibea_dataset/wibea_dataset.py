# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import os
import itertools
from dataclasses import dataclass

import datasets

from learner_datasets.dataset_info import DatasetInfo
from learner_datasets.split_info import SplitInfo


@dataclass
class WIBEAInfo(DatasetInfo):
    CPATH = os.getenv("CORPORA")

    folder = "wi+locness"
    DATA_DIR = os.path.join(CPATH, folder, "orig")
    REF_DIR = os.path.join(CPATH, folder, "m2")
    COR_DIR = os.path.join(CPATH, folder, "cor")
    splits = {
        "train5": SplitInfo(
            file_path=os.path.join(DATA_DIR, "ABC.train.detok.n5.orig"),
            ref_file_path=os.path.join(REF_DIR, "ABC.train.auto.n5.m2"),
            cor_file_path=os.path.join(COR_DIR, "ABC.train.detok.n5.cor"),
        ),
        "train20": SplitInfo(
            file_path=os.path.join(DATA_DIR, "ABC.train.detok.n20.orig"),
            ref_file_path=os.path.join(REF_DIR, "ABC.train.auto.n20.m2"),
            cor_file_path=os.path.join(COR_DIR, "ABC.train.detok.n20.cor"),
        ),
        "train50": SplitInfo(
            file_path=os.path.join(DATA_DIR, "ABC.train.detok.n50.orig"),
            ref_file_path=os.path.join(REF_DIR, "ABC.train.auto.n50.m2"),
            cor_file_path=os.path.join(COR_DIR, "ABC.train.detok.n50.cor"),
        ),
        "train200": SplitInfo(
            file_path=os.path.join(DATA_DIR, "ABC.train.detok.n200.orig"),
            ref_file_path=os.path.join(REF_DIR, "ABC.train.auto.n200.m2"),
            cor_file_path=os.path.join(COR_DIR, "ABC.train.detok.n200.cor"),
        ),
        "train": SplitInfo(
            file_path=os.path.join(DATA_DIR, "ABC.train.detok.orig"),
            ref_file_path=os.path.join(REF_DIR, "ABC.train.auto.m2"),
            cor_file_path=os.path.join(COR_DIR, "ABC.train.detok.cor"),
        ),
        "dev": SplitInfo(
            file_path=os.path.join(DATA_DIR, "ABCN.dev.detok.orig"),
            ref_file_path=os.path.join(REF_DIR, "ABCN.dev.auto.m2"),
            cor_file_path=os.path.join(COR_DIR, "ABCN.dev.detok.cor"),
        ),
    }


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{bryant2019bea,
  title={The BEA-2019 shared task on grammatical error correction},
  author={Bryant, Christopher and Felice, Mariano and Andersen, {\O}istein E and Briscoe, Ted},
  booktitle={Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications},
  pages={52--75},
  year={2019}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
The Write and Improve + LOCNESS dataset for token classification
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
    "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """A conll version of the W&I+LOCNESS dataset"""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    # BUILDER_CONFIGS = [
    #     datasets.BuilderConfig(name="first_domain", version=VERSION, description="This part of my dataset covers a first domain"),
    #     datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    # ]

    # DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = datasets.Features({"sentence": datasets.Value("string")})

        # self.dataset_name = 'wibea_dataset'

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            # filepaths=self.file_paths,
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)

        split_generators = []
        for split, split_info in WIBEAInfo.splits.items():
            split_generators.append(
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "filepath": split_info.file_path,
                        "split": split,
                    },
                )
            )
        return split_generators

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        _id = 0

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                yield _id, {"sentence": line.strip()}

                _id += 1
