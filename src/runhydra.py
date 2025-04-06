import glob
import logging
import os
import time

import jsonlines
from datasets import load_dataset
from dotenv import load_dotenv

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(BASEDIR, ".env"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from transformers import set_seed

from conf import (
    Config,
    CoNLL14DatasetConfig,
    FCEDatasetConfig,
    JFLEGDatasetConfig,
    ParamConfig,
    WIBEADatasetConfig,
    WIFCEDatasetConfig,
    TGIConfig,
    CohereConfig,
    HFConfig,
    OpenAIConfig,
)
from models import get_model, MODEL_TYPE
from utils.prompt_utils import get_prompt_manager
from utils.gpu_utils import print_gpu_utilization
from utils.io_utils import make_folder
from generator import LLMGenerator
from tgi_server import TGIServer

# hydra structured config setup
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

cs.store(group="model", name="tgi_schema", node=TGIConfig)
cs.store(group="model", name="hf_schema", node=HFConfig)
cs.store(group="model", name="cohere_schema", node=CohereConfig)
cs.store(group="model", name="openai_schema", node=OpenAIConfig)

cs.store(group="model/params", name="params_schema", node=ParamConfig)
cs.store(group="dataset", name="wibea_schema", node=WIBEADatasetConfig)
cs.store(group="dataset", name="conll_schema", node=CoNLL14DatasetConfig)
cs.store(group="dataset", name="fce_schema", node=FCEDatasetConfig)
cs.store(group="dataset", name="jfleg_schema", node=JFLEGDatasetConfig)
cs.store(group="dataset", name="wifce_schema", node=WIFCEDatasetConfig)

logger = logging.getLogger(__name__)


def check_and_setup_run_directory(
    base_dir, run_prefix, completion_file, hyp_progress_file
):
    """
    Checks if a run directory exists and if the last run is complete.
    If the last run is complete, creates a new run directory.
    If the last run is not complete:
        - uses the last run directory
        - reads the progress from the last run
    """

    # check how many runs exist within outdir
    existing_folders = glob.glob(f"{base_dir}/{run_prefix}*")

    # check for previous run completion, assume it is there
    completed_last_run = True
    completed_indices = set()
    if len(existing_folders) > 0:
        completion_file = os.path.join(existing_folders[-1], completion_file)
        if not os.path.exists(completion_file):
            completed_last_run = False

    if completed_last_run:
        sub_dir = os.path.join(
            base_dir,
            f"{run_prefix}{len(existing_folders)+1}",
        )
        make_folder(sub_dir)
    else:
        # if last run is not complete,
        # set sub_dir to last run
        sub_dir = os.path.join(
            base_dir,
            f"{run_prefix}{len(existing_folders)}",
        )

        # read progress from previous hypothesis file and set it as the
        # starting point for the current runs
        hyp_path = os.path.join(sub_dir, hyp_progress_file)
        if os.path.exists(hyp_path):
            with jsonlines.open(hyp_path, "r") as f:
                for line in f:
                    completed_indices.add(int(line["sentence_id"]))

        logger.info("Found previous unfinished run")

    return sub_dir, completed_indices


def read_few_shot_sample(file: str, indices: list):
        """
        Read few-shot GEC examples from a file.
        Assumes file is a text file with where each line has the format:
        {original-sentence} ||| {corrected-sentence}
        Args:
            file: file to read from
            indices: indices of examples to read
        """
        few_shot_examples = []
        with open(file, "rt") as f:
            for i, line in enumerate(f):
                if i in indices:
                    sentence, corrected = line.strip().split("|||")
                    few_shot_examples.append((sentence, corrected))
                if len(few_shot_examples) == len(indices):
                    break

        return few_shot_examples


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(args):
    
    t0 = time.time()

    hconf = HydraConfig.get()

    out_dir = hconf.run.dir
    if hconf.mode == RunMode.MULTIRUN:
        out_dir = os.path.join(hconf.sweep.dir, hconf.sweep.subdir)

    logger.info(f"Model output directory: {out_dir}")

    # if config path is provided, load it for re-runs
    if args.config_path is not None:
        args = OmegaConf.load(args.config_path)

    set_seed(args.seed)
    
    logger.info(OmegaConf.to_yaml(args, resolve=True))

    if args.dry_run:
        logger.info("dry_run set to True. Exiting...")
        exit()

    ##################
    # Unload arguments
    ##################
    dataset_name = args.dataset.name.name
    eval_name = args.dataset.split.name
    model_name = args.model.name.name
    model_type = args.model.type

    ##################
    # File IO
    ##################
    # file names
    HYP_FNAME = "hyp.txt"
    HYP_PROGRESS_FNAME = "hyp_progress.jsonl"
    COMPLETION_FNAME = "completion.txt"
    INFO_FNAME = "info.jsonl"
    FAILURE_FNAME = "fail.jsonl"

    sub_dir, completed_indices = check_and_setup_run_directory(
        base_dir=out_dir,
        run_prefix=args.run_prefix,
        completion_file=COMPLETION_FNAME,
        hyp_progress_file=HYP_PROGRESS_FNAME,
    )

    # set new completion file directory
    completion_file = os.path.join(sub_dir, COMPLETION_FNAME)

    logger.info(f"Current run directory: {sub_dir}")

    # save original config for re-runs
    config_file_path = os.path.join(out_dir, "config.yaml")
    with open(config_file_path, "wt") as f:
        OmegaConf.save(args, f, resolve=True)

    ##################
    # load dataset
    ##################
    eval_dataset = load_dataset(
        f"src/learner_datasets/{dataset_name}",
        split=eval_name,
    )

    ##################
    # load prompt manager
    ##################
    prompt_manager_class = get_prompt_manager(model_type)
    prompt_manager = prompt_manager_class(
        model_name=model_name,
        len_multiplier=args.len_multi,
    )
    template = prompt_manager.get_prompt(args.prompt_type, args.prompt_index)

    # check if we have a retrieval file
    if args.retrieval_file is not None and args.retrieval_index is not None:
        indices = [int(i) for i in args.retrieval_index.split(",")]
        retrieval_examples = read_few_shot_sample(args.retrieval_file, indices)
        template = prompt_manager.format_template(
            template,
            retrieval_examples,
        )

    logger.info(f"Prompt template: {template}")

    ##################
    # instantiate params
    ##################
    params = instantiate(args.model.params)

    ##################
    # load model
    ##################
    model_class = get_model(model_type)

    if model_type == MODEL_TYPE.TGI:

        if args.model.manage_server:
            tgi_server = TGIServer(
                model_name=model_name,
                port=args.model.port,
                endpoint=args.model.endpoint)
            try:
                tgi_server.start()
            except Exception as e:
                logger.info("TGI server failed to start")
                logger.info(e)
                tgi_server.stop()
                exit()
        else:
            logger.info("manage_server is set to False.")
            logger.info("Assuming TGI server is running on \
                        endpoint {args.model.endpoint} with \
                        port {args.model.port}.")

        model = model_class(
            model_name=model_name,
            output_dir=sub_dir,
            port=args.model.port,
            endpoint=args.model.endpoint
        )
    else:
        model = model_class(
            model_name=model_name,
            output_dir=sub_dir,
        )

    print_gpu_utilization()

    # save modified config with model type and prompt template
    # add runtime parameters
    with open_dict(args):
        args.prompt_template = template

    config_file_path = os.path.join(out_dir, "config_detailed.yaml")
    with open(config_file_path, "wt") as f:
        OmegaConf.save(args, f, resolve=True)

    ##################
    # setup llm generator
    ##################

    # setup paths
    model_hyp_file = os.path.join(sub_dir, HYP_FNAME)
    hyp_progress_file = os.path.join(sub_dir, HYP_PROGRESS_FNAME)
    info_file = os.path.join(sub_dir, INFO_FNAME)
    failure_file = os.path.join(sub_dir, FAILURE_FNAME)

    llm_generator = LLMGenerator(
        model=model,
        params=params,
        prompt_manager=prompt_manager,
        prompt_template=template,
        eval_dataset=eval_dataset,
        batch_size=args.batch,
        sub_dir=sub_dir,
        model_hyp_file=model_hyp_file,
        hyp_progress_file=hyp_progress_file,
        info_file=info_file,
        failure_file=failure_file,
        completed_indices=completed_indices,
        verbose=args.verbose,
    )

    ##################
    # generate output
    ##################
    try:
        successes, failures = llm_generator.run()
    finally:
        # optionally stop TGI server
        if model_type == MODEL_TYPE.TGI:
            if args.model.manage_server:
                if tgi_server is not None:
                    tgi_server.stop()
            else:
                logger.info("manage_server is set to False, and therefore \
                            this script will not stop the TGI server.")
            

    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")

    ##################
    # wrap up
    ##################

    if failures > 0:
        logger.info(f"Failed on {failures} sentences")
        logger.info(f"See {failure_file} for details")
        logger.info("Exiting...")
        exit()
    else:
        # read sentences from hyp_progress_file and write to model_hyp_file
        #   using sentence_id as the order
        hyp_sentences = {}
        with jsonlines.open(hyp_progress_file, "r") as f:
            for line in f:
                sentence_id = line["sentence_id"]
                output = line["output"]
                hyp_sentences[sentence_id] = output

        with open(model_hyp_file, "wt") as f:
            for sentence_id in sorted(hyp_sentences.keys()):
                f.write(hyp_sentences[sentence_id] + "\n")

    # save completion file
    t1 = time.time()
    completed_in = t1 - t0
    # convert seconds to HH:MM:SS
    completed_in = time.strftime("%H:%M:%S", time.gmtime(completed_in))
    logger.info(f"Completed in: {completed_in}")
    with open(completion_file, "wt") as f:
        f.write(f"Completed in: {completed_in}")

    # free up gpu memory explicitly
    del model
    # delete hyp_progress_file
    os.remove(hyp_progress_file)


if __name__ == "__main__":
    main()
