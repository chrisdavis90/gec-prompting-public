import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
load_dotenv(os.path.join(BASEDIR, ".env"))

# add the src directory to the path
sys.path.append(os.path.join(BASEDIR, "src"))

from evaluate_gleu import gleu_evaluation_main
from learner_datasets import get_dataset_info
from reporting import create_results_report, progress_few_shot, progress_zero_shot
from evaluate_conll14 import main as m2scorer_main

logger = logging.getLogger(__name__)


def errant_result_file_name(results_tag, hyp_name, errant_tag):
    results_file_parts = [results_tag, hyp_name]
    if errant_tag:
        results_file_parts.append(errant_tag)
    results_file_name = "_".join(results_file_parts)
    results_file = f"{results_file_name}.json"

    return results_file


def format_sentence_info(sentences):
    """Format sentence info from errant compare output"""
    data = []

    # groups of three lines, indexed by sentence number
    sentence_groups = defaultdict(dict)
    original = "original sentence"
    edits = "edits"
    additional = "additional info"

    previous_component = None
    previous_sentence_index = None

    # iterate through sentences
    for sentence in sentences:
        # original sentence
        if sentence.startswith("\nOriginal sentence "):
            # "\nOriginal sentence 554: It 's more common eat pizza .\n"
            sentence_index = sentence.split(":")[0]
            sentence_index = int(sentence_index.split(" ")[-1])

            if sentence_index not in sentence_groups:
                sentence_groups[sentence_index] = {}
            if original not in sentence_groups[sentence_index]:
                sentence_groups[sentence_index].update({original: []})

            sentence_groups[sentence_index][original].append(sentence.strip())
            previous_component = original
            previous_sentence_index = sentence_index

        elif sentence.startswith("\nSENTENCE "):
            # \nSENTENCE 554 - HYP 0 - REF 0\nHYPOTHESIS EDITS :
            sentence_index = sentence.split("-")[0].strip()
            sentence_index = int(sentence_index.split(" ")[-1])

            if sentence_index not in sentence_groups:
                sentence_groups[sentence_index] = {}
            if edits not in sentence_groups[sentence_index]:
                sentence_groups[sentence_index].update({edits: []})

            sentence_groups[sentence_index][edits].append(sentence.strip())
            previous_component = edits
            previous_sentence_index = sentence_index

        elif sentence.startswith("\n^^ HYP 0, REF "):
            # \n^^ HYP 0, REF 0 chosen for sentence 554\nLocal results:
            sentence_index = sentence.split("\n")[1]
            sentence_index = int(sentence_index.split(" ")[-1])

            if sentence_index not in sentence_groups:
                sentence_groups[sentence_index] = {}
            if additional not in sentence_groups[sentence_index]:
                sentence_groups[sentence_index].update({additional: []})

            sentence_groups[sentence_index][additional].append(sentence)
            previous_component = additional
            previous_sentence_index = sentence_index
        else:
            sentence_groups[previous_sentence_index][previous_component] += sentence

    for sentence_index, sentence_group in sentence_groups.items():
        sentence_data = {}

        sentence_data["sentence_index"] = sentence_index

        # process original sentence
        original_sentence = sentence_group[original][0]
        sentence_data["original_sentence"] = original_sentence.split(":", maxsplit=1)[1]

        # process hypothesis edits
        sentence_data["annotation_info"] = {}
        for edit_group in sentence_group[edits]:
            sentence_edits = {}
            title = None
            for line in edit_group.split("\n"):
                if ":" in line:
                    key, value = line.split(":", maxsplit=1)
                    sentence_edits[key.strip()] = value.strip()
                else:
                    if len(line.strip()) > 0:
                        # sentence_edits["annotation_info"] = line.strip()
                        title = line.strip()

            sentence_data["annotation_info"][title] = sentence_edits

        additional_info = sentence_group[additional][0]
        sentence_data["additional_info"] = additional_info

        data.append(sentence_data)

    return data


def errant_evaluate(
    sub_dir: str,
    hyp_name: str = "hyp_post",
    results_tag: str = "results",
    errant_tag: str = None,
    n: int = 50,
    force: bool = False,
    tokenise: bool = True,
    levenstein: bool = False,
    verbose: bool = True,
):
    """Evaluate a single experiment folder"""
    base_folder = os.path.dirname(sub_dir)
    config_path = os.path.join(base_folder, "config.yaml")

    # hypothesis file with model generations
    model_hyp_file = os.path.join(sub_dir, f"{hyp_name}.txt")

    # output file to save results
    results_file_name = errant_result_file_name(results_tag, hyp_name, errant_tag)
    results_file = os.path.join(sub_dir, f"{results_file_name}")
    if not force and os.path.exists(results_file):
        return

    args = OmegaConf.load(config_path)

    dataset_name = args.dataset.name
    eval_name = args.dataset.split

    dataset_info = get_dataset_info(dataset_name)

    ##################
    # Run errant parallel
    ##################
    source_file = dataset_info.splits[eval_name].file_path
    out_m2_parts = [hyp_name]
    if errant_tag:
        out_m2_parts.append(errant_tag)
    out_m2_file_name = "_".join(out_m2_parts)
    out_m2 = os.path.join(sub_dir, f"{out_m2_file_name}.m2")
    errant_parallel_args = [
        "errant_parallel",
        "-orig",
        source_file,
        "-cor",
        model_hyp_file,
        "-out",
        out_m2,
    ]
    if tokenise:
        errant_parallel_args.append("-tok")
    if levenstein:
        errant_parallel_args.append("-lev")

    subprocess.run(errant_parallel_args)

    ##################
    # Run errant compare
    ##################
    ref_file = dataset_info.splits[eval_name].ref_file_path
    errant_compare_args = [
        "errant_compare",
        "-hyp",
        out_m2,
        "-ref",
        ref_file,
        # "-cat",
        # "3",
    ]
    if verbose:
        errant_compare_args.append("-v")
    stdoutput = subprocess.check_output(errant_compare_args).decode("utf-8")

    if verbose:
        evaluation_components = stdoutput.split(
            "=========== Span-Based Correction ============"
        )

        sentence_info = evaluation_components[0].split(
            "----------------------------------------"
        )
        sentence_info = format_sentence_info(sentence_info[1:])

        rows = []
        for info in sentence_info:
            slen = len(info["original_sentence"].split())
            for _, edit_info in info["annotation_info"].items():
                n_ref_edits = len(eval(edit_info["REFERENCE EDITS"]))
                local_p, local_r, local_score = (float(x) for x in edit_info["Local P/R/F0.5"].split())
                local_tp, local_fp, local_fn = (float(x) for x in edit_info["Local TP/FP/FN"].split())

                rows.append([slen, n_ref_edits, local_p, local_r, local_score, local_tp, local_fp, local_fn])

        # create dataframe
        df = pd.DataFrame(rows, columns=["length", "n_ref_edits", "local_p", "local_r", "local_score", "local_tp", "local_fp", "local_fn"])
        # save dataframe
        df.to_csv(os.path.join(sub_dir, "sentence_results.csv"), index=False)

        # filter sentence info to first n sentences
        if n is not None and n > 0:
            sentence_info = sentence_info[:n]

        per_error_results = evaluation_components[1].split("\n")[1:-2]
        # list to pandas dataframe
        per_error_results = [x.split() for x in per_error_results]
        df = pd.DataFrame(per_error_results)
        # make first row the header and drop it
        df.columns = df.iloc[0]
        df = df.drop(0)
        # dataframe to json object

        overall_results = evaluation_components[-1].split("\n")
        header = overall_results[1].split("\t")
        values = overall_results[2].split("\t")
        results = dict(zip(header, values))

        with open(results_file, "wt") as f:
            json.dump(results, f, indent=4)
        print(
            args.model,
            "\t",
            f"{args.prompt_type}-{args.prompt_index}",
            "\t",
            f"{args.dataset.name}-{args.dataset.split}",
            "\t",
            results["F0.5"],
        )

        # save all info
        data = {
            "sentence_results": sentence_info,
            "per_error_results": per_error_results,
            "overall_results": results,
        }
        results_name = os.path.basename(results_file).split(".")[0]
        all_results_file = os.path.join(sub_dir, f"all_{results_name}.json")
        with open(all_results_file, "wt") as f:
            json.dump(data, f, indent=4)
    else:
        stdoutput = stdoutput.split("\n")

        header = stdoutput[2].split("\t")
        values = stdoutput[3].split("\t")
        # create dict from header and values
        results = dict(zip(header, values))

        with open(results_file, "wt") as f:
            json.dump(results, f, indent=4)
        print(
            args.model,
            "\t",
            f"{args.prompt_type}-{args.prompt_index}",
            "\t",
            f"{args.dataset.name}-{args.dataset.split}",
            "\t",
            results["F0.5"],
        )


def errant_evaluation_main(
    base_dir,
    exp_folder,
    exp_pattern,
    hyp_name,
    results_tag,
    errant_tag,
    p,
    n,
    force,
    tokenise,
    levenstein,
    verbose,
    run_pattern="run_1",
):
    folders = glob.glob(
        os.path.join(
            base_dir,
            exp_folder,
            exp_pattern,
            run_pattern,
        )
    )

    print(f"Found {len(folders)} folders for errant evaluation")

    sub_dir_args = []
    missing_completions = []
    for sub_dir in tqdm(folders):
        completion_file = os.path.join(sub_dir, "completion.txt")
        if not os.path.exists(completion_file):
            missing_completions.append(sub_dir)
            continue

        sub_dir_args.append(
            (
                sub_dir,
                hyp_name,
                results_tag,
                errant_tag,
                n,
                force,
                tokenise,
                levenstein,
                verbose,
            )
        )

    if sub_dir_args:
        p = min(p, len(sub_dir_args))
        # p = 1
        with Pool(processes=p) as process_pool:
            process_pool.starmap(
                errant_evaluate,
                tqdm(sub_dir_args, total=len(sub_dir_args)),
            )


def post_process(
    base_dir,
    exp_folder,
    exp_pattern,
    hyp_in_name,
    hyp_out_name,
    force=False,
    run_pattern="run_1",
):
    folders = glob.glob(
        os.path.join(
            base_dir,
            exp_folder,
            exp_pattern,
            run_pattern,
        )
    )
    # filter to only directories
    folders = [x for x in folders if os.path.isdir(x)]
    print(f"Found {len(folders)} folders to post process")

    # llama start tokens derived from these examples:
    # "Here is a revised version of the text with minimal changes to make it grammatically correct:",
    # "Here's a revised version of the text with minimal changes for grammar correction:",
    # "Sure! Here's a revised version of the sentence that's grammatically correct:",
    # "Here's a grammatically corrected version of the text:",
    # "Here's a corrected version of the text:",
    # "Sure! Here's a grammatically corrected version of the text:",
    # "Here is a grammatically corrected version of the text:",
    # "Here's a revised version of the text with minimal changes for grammatical correctness:",
    # "Here's a grammatically corrected version of the text:",
    # "Sure! Here's a corrected version of the sentence with minimal changes:",
    llama_start_tokens = [
        "Sure! Here",
        "Sure! The sentence",
        "Here is a",
        "Here's a",
    ]

    llama_suffix_tokens = [
        "\(No changes",
        "Explanation:",
        "\(The corrections",
        "\(No correction",
        "Corrections:",
        "Is there anything",
        "Here's a list of",
        "Here is a list of",
        "The original sentence",
        "\(The original sentence",
        "In the original sentence",
        "\(The sentence",
        "\(The only error in",
        "\(Changes made:",
        "\(The change made",
        "\(Note: ",
        "The main issue",
        "The only change I made",
        "I changed",
        "I made.*changes",
        "Input sentence: ",
    ]

    missing_run_folders = []
    missing_hyp_files = []
    completed = 0

    for run_folder in tqdm(folders):
        # hypothesis file
        hyp_file = os.path.join(run_folder, hyp_in_name)
        if not os.path.exists(hyp_file):
            missing_hyp_files.append(run_folder)
            continue

        # post_process file
        postproc_file = os.path.join(run_folder, hyp_out_name)
        if not force and os.path.exists(postproc_file):
            continue

        # iterator for lines in hyp_file
        with open(postproc_file, "w") as outf:
            with open(hyp_file, "r") as inf:
                for line in inf:
                    # post_process line
                    line = line.strip()  # remove new-line token

                    # if line starts with "Output sentence: " or "Corrected sentence: ", remove it
                    if line.startswith("Output sentence: "):
                        line = line[len("Output sentence: ") :]
                    elif line.startswith("Corrected sentence: "):
                        line = line[len("Corrected sentence: ") :]
                    elif line.startswith("Input sentence: "):  # for gpt-3.5-turbo
                        line = line[len("Input sentence: ") :]

                    if "Llama" in postproc_file:
                        # llama checks
                        # check if sentence starts with special tokens
                        for prefix in llama_start_tokens:
                            m = re.search(prefix, line)
                            # check if m is found at the beginning of the line
                            if m and m.start() == 0:
                                # split on first occurence of colon
                                # assume that prefixes end with a colon
                                if ":" in line:
                                    line = line.split(":", 1)[1].strip()
                                break

                        # check for suffixes
                        for suffix in llama_suffix_tokens:
                            m = re.search(suffix, line)
                            if m:
                                # split line at the beginning of match
                                line = line[: m.start()].strip()
                                break

                    if "falcon" in postproc_file:
                        # check if "Input sentence:" is in the string
                        if "Input sentence:" in line:
                            # remove everything after and including "Input sentence:"
                            line = line.split("Input sentence:")[0].strip()

                    # if line starts and ends with double quote, remove them
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    # if line is odd and ends with double quote, remove it
                    # count quotations in line:
                    if line.count('"') % 2 == 1 and line.endswith('"'):
                        line = line[:-1]

                    # write line to postproc_file
                    outf.write(line + "\n")

        completed += 1

    print(f"Completed: {completed}")
    # print(f"Missing run folders: {len(missing_run_folders)}")
    print(f"Missing hyp files: {len(missing_hyp_files)}")

    return completed, missing_run_folders, missing_hyp_files


def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        type=str,
        help="base directory",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="folder containing experiments",
        default="output_final",
    )
    parser.add_argument(
        "--exps_pattern",
        type=str,
        help="pattern matching experiments to evaluate",
        default="*",
    )
    parser.add_argument(
        "--run_pattern",
        type=str,
        help="pattern matching run folders to evaluate",
        default="run_1",
    )
    parser.add_argument(
        "--jfleg_pattern",
        type=str,
        help="pattern matching jfleg files",
        default="*jfleg*",
    )
    parser.add_argument(
        "--conll_pattern",
        type=str,
        help="pattern matching conll files",
        default="*conll*",
    )
    parser.add_argument(
        "--hyp",
        type=str,
        help="name of hypothesis file",
        default="hyp",
    )
    parser.add_argument(
        "--hyp_tag",
        type=str,
        help="tag for post-processed hypothesis file",
        default="_post",
    )
    parser.add_argument(
        "--results_tag",
        type=str,
        help="base tag for results files",
        default="results",
    )
    parser.add_argument(
        "--errant_tag",
        type=str,
        help="tag to version errant results files",
        default="errantv2-3-3",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="number of sentences to log in full errant evaluation",
        default=50,
    )
    parser.add_argument(
        "-p",
        type=int,
        help="number of processes to use",
        default=30,
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="force post-processing and evaluation",
    )
    parser.add_argument(
        "-tok",
        action="store_true",
        help="tokenise sentences with errant",
        default=True,
    )
    parser.add_argument(
        "-lev",
        action="store_true",
        help="use levenstein distance for alignment",
        default=False,
    )
    parser.add_argument(
        "--results_report",
        type=str,
        help="output file for results report",
        default="results.csv",
    )

    return parser


def main(args, do_post_process=True, report_results=False, report_progress=False):
    print("Running evaluation script")

    # post-process output files
    completed, missing_run_folders, missing_hyp_files = [], [], []
    if do_post_process:
        completed, missing_run_folders, missing_hyp_files = post_process(
            base_dir=args.dir,
            exp_folder=args.folder,
            exp_pattern=args.exps_pattern,
            hyp_in_name=f"{args.hyp}.txt",
            hyp_out_name=f"{args.hyp}{args.hyp_tag}.txt",
            force=args.force,
            run_pattern=args.run_pattern,
        )

    # print missing run folders
    if len(missing_run_folders) > 0:
        print("Missing run folders:")
        for folder in missing_run_folders:
            print(folder)

    # print missing hyp folders
    if len(missing_hyp_files) > 0:
        print("Missing hyp files:")
        for folder in missing_hyp_files:
            print(folder)

    # run errant evaluation
    errant_evaluation_main(
        base_dir=args.dir,
        exp_folder=args.folder,
        exp_pattern=args.exps_pattern,
        hyp_name=f"{args.hyp}{args.hyp_tag}",
        results_tag=args.results_tag,
        errant_tag=args.errant_tag,
        p=args.p,
        n=args.n,
        force=args.force,
        tokenise=args.tok,
        levenstein=args.lev,
        verbose=True,
        run_pattern=args.run_pattern,
    )

    # run conll14 evaluation
    m2scorer_main(
        base_dir=args.dir,
        exp_folder=args.folder,
        exp_pattern=args.conll_pattern,
        hyp_name=f"{args.hyp}{args.hyp_tag}",
        p=args.p,
        force=args.force,
        run_pattern=args.run_pattern,
    )

    # # run gleu evaluation
    gleu_evaluation_main(
        base_dir=args.dir,
        exp_folder=args.folder,
        exp_pattern=args.jfleg_pattern,
        hyp_name=f"{args.hyp}{args.hyp_tag}",
        p=args.p,
        run_pattern=args.run_pattern,
    )

    if report_results:
        # create report
        errant_file = errant_result_file_name(
            args.results_tag, f"{args.hyp}{args.hyp_tag}", args.errant_tag
        )
        gleu_file = f"results_gleu_{args.hyp}{args.hyp_tag}.json"
        m2scorer_file = f"results_hyp_post_m2scorer.json"
        create_results_report(
            path=os.path.join(args.dir, args.folder),
            errant_file=errant_file,
            gleu_file=gleu_file,
            m2scorer_file=m2scorer_file,
            output=args.results_report,
        )

    if report_progress:
        progress_zero_shot(
            errant_result_file=errant_file,
            gleu_result_file=gleu_file,
        )
        progress_few_shot(
            errant_result_file=errant_file,
            gleu_result_file=gleu_file,
        )


def evaluate_output_final_zero_shot():
    parser = create_args()

    # parse with overrides
    args = parser.parse_args(
        [
            "--dir",
            "paper_output",
            "--folder",
            "output_zero_shot_dev",
            # "--force",
        ]
    )

    main(args, report_results=True)


def evaluate_output_filter():
    parser = create_args()

    # parse with overrides
    args = parser.parse_args(
        [
            "--folder",
            "output_prompt_filter",
            "--force",
        ]
    )

    main(args, report_results=True)


def evaluate_output_final_few_shot():
    parser = create_args()

    # parse with overrides
    args = parser.parse_args(
        [
            "--dir",
            "paper_output",
            "--folder",
            "output_few_shot_dev",
            # "--force",
        ]
    )

    main(args, report_results=True)


def evaluate_test():
    parser = create_args()

    # parse with overrides
    args = parser.parse_args(
        [
            "--dir",
            "paper_output",
            "--folder",
            "output_test",
            "--exps_pattern",
            "*",
            "--force",
            "--run_pattern",
            "run_*",
        ]
    )

    main(args, report_results=True)


def evaluate_llamas():
    parser = create_args()

    # parse with overrides
    args = parser.parse_args(
        [
            "--folder",
            "output_llama/*",
            "--exps_pattern",
            "*70b*",
            "--force",
            "--run_pattern",
            "run_*",
        ]
    )

    main(args)


if __name__ == "__main__":
    # evaluate_output_final()
    evaluate_output_final_few_shot()
    # evaluate_output_filter()
    # evaluate_llamas()
    # evaluate_test()

    # parser = create_args()
    # args = parser.parse_args()
    # main(args)
