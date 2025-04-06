import logging
import os
import re
import time

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(BASEDIR, ".env"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(
        self,
        model,
        params,
        prompt_manager,
        prompt_template,
        eval_dataset,
        batch_size,
        sub_dir,
        model_hyp_file,
        hyp_progress_file,
        info_file,
        failure_file,
        completed_indices=list(),
        verbose=False,
    ):
        self.sub_dir = sub_dir
        self.model_hyp_file = model_hyp_file
        self.hyp_progress_file = hyp_progress_file
        self.info_file = info_file
        self.failure_file = failure_file

        self.eval_dataset = eval_dataset
        self.model = model
        self.prompt_manager = prompt_manager
        self.template = prompt_template
        self.params = params

        self.completed_indices = completed_indices
        self.batch_size = batch_size
        self.verbose = verbose

    def run(self):
        t0 = time.time()
        sentence_counter = 0

        batch = []
        batch_info = {}

        dataset_iter = iter(KeyDataset(self.eval_dataset, "sentence"))

        successes = 0
        failures = 0

        pbar = tqdm(total=len(self.eval_dataset))

        while True:
            # create batch
            while len(batch) < self.batch_size:
                try:
                    text = next(dataset_iter)

                    if sentence_counter in self.completed_indices:
                        sentence_counter += 1
                        pbar.update(1)
                        continue

                    input_prompt = self.prompt_manager.format_prompt(
                        self.template, text
                    )
                    template_len, text_len = self.model.count_tokens(
                        self.template, text
                    )
                    prompt_len = template_len + text_len
                    max_len = int(
                        prompt_len + (text_len * self.prompt_manager.len_multiplier)
                    )
                    self.params.update_max_len(max_len)

                    # estimated usage
                    info = {
                        "sentence_id": sentence_counter,
                        "prompt_tokens": prompt_len,
                        "input_tokens": text_len,
                        "estimated_max_len": max_len,
                    }

                    batch.append(
                        {
                            "sentence_id": sentence_counter,
                            "prompt": input_prompt,
                            "kwargs": vars(self.params).copy(),
                        }
                    )
                    batch_info[sentence_counter] = info

                    sentence_counter += 1
                except StopIteration:
                    # end of dataset
                    break

            # if batch is empty, end of dataset
            if len(batch) == 0:
                break

            # process batch
            output_batch = self.model.run_batch(prompt_info_batch=batch)

            # iterate through sorted output_batch
            for sentence_id, response_obj in sorted(output_batch.items()):
                output = response_obj.output
                errors = response_obj.errors

                success = output is not None

                if success:
                    # Post-processing (could be injected)
                    # replace newlines from string with a space
                    # we do this here instead of a post-proc step to ensure
                    #  the output is one line per sentence
                    output = output.replace("\n", " ")
                    # regex to replace multiple spaces with single space
                    output = re.sub(r"\s+", " ", output)

                    if self.verbose:
                        logger.info(f"Original: {batch[sentence_id]}")
                        logger.info(f"Hypothesis: {output}")

                    # write to file
                    with jsonlines.open(self.hyp_progress_file, "a") as f:
                        f.write_all([{"sentence_id": sentence_id, "output": output}])

                    # save info as jsonlines
                    batch_info[sentence_id]["duration"] = response_obj.duration
                    with jsonlines.open(self.info_file, "a") as f:
                        f.write_all([batch_info[sentence_id]])
                    successes += 1
                else:
                    # report failures
                    with jsonlines.open(self.failure_file, "a") as f:
                        f.write_all([{sentence_id: errors}])
                    failures += 1
                    logger.info(
                        f"Failed on sentence: {sentence_id}: {batch[sentence_id]}"
                    )

            # reset batch
            pbar.update(len(batch))
            batch = []
            batch_info = {}
            self.model.sleep()

        t1 = time.time()
        # convert seconds to HH:MM:SS
        completed_in = t1 - t0
        completed_in = time.strftime("%H:%M:%S", time.gmtime(completed_in))
        logger.info(f"Processed sentences in: {completed_in}")

        return successes, failures
