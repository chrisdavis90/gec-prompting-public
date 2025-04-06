from dataclasses import dataclass
from enum import Enum
import os


@dataclass
class DatasetInfo:
    # load CORPORA from environment variable
    CPATH = os.getenv("CORPORA")
    DATA_DIR: str
    REF_DIR: str
    COR_DIR: str
    splits: dict
    SplitsEnum: Enum
