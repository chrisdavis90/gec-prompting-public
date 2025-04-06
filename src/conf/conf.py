from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from omegaconf import MISSING, SI, OmegaConf

from learner_datasets import (
    CoNLL14SplitsEnum,
    DatasetEnum,
    FCESplitsEnum,
    JFLEGSplitsEnum,
    WIBEASplitsEnum,
    WIFCESplitsEnum,
)
from models import ModelListEnum, MODEL_TYPE


# helper functions to process config variables
def to_string_resolver(x):
    return str(x)


def model_name_resolver(x: ModelListEnum):
    return x.name.split("/")[-1]


def prompt_type_resolver(x):
    return x.replace("-", "_")


def to_enum_name(x):
    return x.name


OmegaConf.register_new_resolver("str", to_string_resolver)
OmegaConf.register_new_resolver("mstr", model_name_resolver)
OmegaConf.register_new_resolver("pstr", prompt_type_resolver)
OmegaConf.register_new_resolver("enumstr", to_enum_name)


@dataclass
class ParamConfig:
    """
    HF Pipeline Params:
    # temperature: float
    # # max_new_tokens: int = -1
    # num_return_sequences: int
    # return_full_text: bool
    # do_sample: bool
    # top_k: int
    # top_p: float
    # num_beams: int

    OpenAI Params:
    temperature: float
    top_p: float

    Cohere Params:
    temperature: float
    num_generations: int
    return_likelihoods: str
    k: int
    p: float
    """

    gen_kwargs: dict
    name: str
    _target_: str


@dataclass
class BaseDatasetConfig:
    name: DatasetEnum
    start_index: int
    end_index: int
    split: Any = MISSING


@dataclass
class WIBEADatasetConfig(BaseDatasetConfig):
    split: WIBEASplitsEnum


@dataclass
class CoNLL14DatasetConfig(BaseDatasetConfig):
    split: CoNLL14SplitsEnum


@dataclass
class FCEDatasetConfig(BaseDatasetConfig):
    split: FCESplitsEnum


@dataclass
class JFLEGDatasetConfig(BaseDatasetConfig):
    split: JFLEGSplitsEnum


@dataclass
class WIFCEDatasetConfig(BaseDatasetConfig):
    split: WIFCESplitsEnum


@dataclass
class ModelConfig:
    name: ModelListEnum = MISSING
    type: MODEL_TYPE = MISSING
    params: Any = MISSING

@dataclass
class HFConfig(ModelConfig):
    type: MODEL_TYPE = MODEL_TYPE.HF    

@dataclass
class CohereConfig(ModelConfig):
    type: MODEL_TYPE = MODEL_TYPE.COHERE

@dataclass
class TGIConfig(ModelConfig):
    type: MODEL_TYPE = MODEL_TYPE.TGI
    port: int = 8080
    endpoint: str = "http://localhost"
    health_endpoint: str = "/health"
    manage_server: bool = False

@dataclass 
class OpenAIConfig(ModelConfig):
    type: MODEL_TYPE = MODEL_TYPE.OPENAI



defaults = [
    {"params": "hf"},
    {"dataset": "wibea"},
]


@dataclass
class Config:
    # defaults: List[Any] = field(default_factory=lambda: defaults)

    base_dir: str = MISSING
    out_dir: str = MISSING
    run_prefix: str = MISSING

    # prompt
    prompt_type: str = MISSING
    prompt_index: int = MISSING
    len_multi: float = 1.5
    retrieval_file: str = MISSING
    retrieval_index: str = MISSING

    verbose: bool = False
    batch: int = MISSING
    seed: int = MISSING
    dry_run: bool = False

    config_path: Optional[str] = None  # optional path to load config from
    model: ModelConfig = MISSING
    dataset: BaseDatasetConfig = MISSING

    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "${out_dir}"},
            "sweep": {"dir": "${out_dir}", "subdir": ""},
            "job": {"chdir": False},
            "verbose": False,  # True
        }
    )
