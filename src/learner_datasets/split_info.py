from typing import Union
from dataclasses import dataclass


@dataclass
class SplitInfo:
    file_path: str
    ref_file_path: str
    cor_file_path: Union[str, list[str]]
