import os
from dataclasses import dataclass
from enum import Enum
from typing import Union

from .split_info import SplitInfo
from .dataset_info import DatasetInfo
from .conll14_dataset.conll14_dataset import CoNLL14Info
from .fce_dataset.fce_dataset import FCEInfo
from .jfleg_dataset.jfleg_dataset import JFLEGInfo
from .wibea_dataset.wibea_dataset import WIBEAInfo
from .wifce_dataset.wifce_dataset import WIFCEInfo


class DatasetEnum(Enum):
    wibea_dataset = "wibea_dataset"
    conll14_dataset = "conll14_dataset"
    jfleg_dataset = "jfleg_dataset"
    fce_dataset = "fce_dataset"
    wifce_dataset = "wifce_dataset"


WIBEASplitsEnum = Enum(
    "WIBEASplitsEnum", {split: i for i, split in enumerate(WIBEAInfo.splits.keys())}
)
FCESplitsEnum = Enum(
    "FCESplitsEnum", {split: i for i, split in enumerate(FCEInfo.splits.keys())}
)
JFLEGSplitsEnum = Enum(
    "JFLEGSplitsEnum", {split: i for i, split in enumerate(JFLEGInfo.splits.keys())}
)
CoNLL14SplitsEnum = Enum(
    "CoNLL14SplitsEnum", {split: i for i, split in enumerate(CoNLL14Info.splits.keys())}
)
WIFCESplitsEnum = Enum(
    "WIFCESplitsEnum", {split: i for i, split in enumerate(WIFCEInfo.splits.keys())}
)


def get_dataset_info(dataset: Union[DatasetEnum, str]):
    if type(dataset) == str:
        dataset = DatasetEnum[dataset]

    if dataset == DatasetEnum.wibea_dataset:
        return WIBEAInfo
    elif dataset == DatasetEnum.conll14_dataset:
        return CoNLL14Info
    elif dataset == DatasetEnum.jfleg_dataset:
        return JFLEGInfo
    elif dataset == DatasetEnum.fce_dataset:
        return FCEInfo
    elif dataset == DatasetEnum.wifce_dataset:
        return WIFCEInfo
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
