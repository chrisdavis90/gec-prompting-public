import os
import glob
import subprocess
import sys
import json
from multiprocessing import Pool
from tqdm import tqdm
import time

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# add the base directory to the path
sys.path.append(os.path.join(BASEDIR))

from lib.m2scorer.scripts.Tokenizer import PTBTokenizer


def tokenize(infile, outfile):
    assert os.path.exists(infile)

    tokenizer = PTBTokenizer()

    with open(outfile, "wt") as outf:
        with open(infile, "r") as f:
            for line in f:
                tokens = tokenizer.tokenize(line.strip())
                out = " ".join(tokens)
                outf.write(out + "\n")


def rename_metric(name):
    # rename to match ERRANT naming
    if name == "Precision":
        return "Prec"
    elif name == "Recall":
        return "Rec"
    else:
        return "F0.5"


def evaluate(hyp_file, hyp_name, force=False):
    assert os.path.exists(hyp_file)
    folder = os.path.dirname(hyp_file)

    gold_file = os.path.join(os.environ.get('CORPORA'), "conll2014/m2/conll2014.2.test.auto.m2")
    output_file = os.path.join(folder, f"results_{hyp_name}_m2scorer.json")

    if not force and os.path.exists(output_file):
        return

    start = time.time()
    print("Evaluating: ", os.path.basename(os.path.dirname(folder)))

    m2scorer_path = os.path.join(BASEDIR, "lib", "m2scorer", "scripts", "m2scorer.py")
    args = [
        "python",
        m2scorer_path,
        os.path.abspath(hyp_file),
        gold_file,
    ]

    ####

    try:
        stdout = subprocess.check_output(
            args, cwd=os.path.join(BASEDIR, "lib", "m2scorer", "scripts")
        ).decode("utf8")

    except subprocess.CalledProcessError as e:
        print(e)
        print(e.output)
        raise e

    ####

    # null stats because the m2scorer doesn't return them
    results = {
        "TP": "0",
        "FP": "0",
        "FN": "0",
    }

    lines = stdout.split("\n")
    for line in lines:
        if len(line.strip()) == 0:
            continue

        metric, value = line.split(":")
        metric = rename_metric(metric.strip())
        value = value.strip()
        results[metric] = value

    with open(output_file, "wt") as f:
        json.dump(results, f, indent=4)

    end = time.time()
    # duration in MM:SS
    duration = time.strftime("%M:%S", time.gmtime(end - start))
    print(f"Finished {os.path.basename(os.path.dirname(folder))} in {duration} (MM:SS)")


def main(
    base_dir, exp_folder, exp_pattern, hyp_name, p=1, run_pattern="run_1", force=False
):
    folders = glob.glob(os.path.join(base_dir, exp_folder, exp_pattern, run_pattern))
    if len(folders) == 0:
        return

    # remove folders that don't contain "conll"
    folders = [f for f in folders if "conll" in f]

    input_files, output_files = [], []

    for folder in folders:
        hyp_file = os.path.join(folder, f"{hyp_name}.txt")
        input_files.append(hyp_file)
        output_files.append(os.path.join(folder, f"{hyp_name}_tok.txt"))

    # tokenizing
    if input_files:
        p = min(p, len(input_files))
        with Pool(processes=p) as process_pool:
            process_pool.starmap(
                tokenize,
                tqdm(zip(input_files, output_files), total=len(input_files)),
            )

    # m2scorer
    output_files = [(x, hyp_name, force) for x in output_files]
    if input_files and output_files:
        p = min(p, len(output_files))
        with Pool(processes=p) as process_pool:
            process_pool.starmap(
                evaluate,
                tqdm(output_files, total=len(output_files)),
            )


if __name__ == "__main__":
    base_dir = ""
    exp_folder = "output_final"
    exp_pattern = "conll14*bloomz*zero_shot*"
    run_pattern = "run_1"
    hyp_name = "hyp_post"

    min_processes = 10

    main(
        base_dir,
        exp_folder,
        exp_pattern,
        hyp_name,
        p=min_processes,
        run_pattern=run_pattern,
    )
