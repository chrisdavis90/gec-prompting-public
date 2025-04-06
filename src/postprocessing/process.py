import os
import glob
import sys
from tqdm import tqdm

if __name__ == "__main__":
    base = "/path/to/projects/gec-prompting/output_prompt_filter"

    for folder in tqdm(
        glob.glob(
            os.path.join(
                base,
                "*",
            )
        )
    ):
        run_folder = os.path.join(folder, "run_1")
        if not os.path.exists(run_folder):
            continue

        # hypothesis file
        hyp_file = os.path.join(run_folder, "hyp.txt")
        if not os.path.exists(hyp_file):
            continue

        print(run_folder)

        # post_process file
        postproc_file = os.path.join(run_folder, "hyp_post_v2.txt")
        # if os.path.exists(postproc_file):
        #     continue

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

                    # if line starts and ends with double quote, remove them
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]

                    # if line is odd and ends with double quote, remove it
                    # count quotations in line:
                    if line.count('"') % 2 == 1 and line.endswith('"'):
                        line = line[:-1]

                    # write line to postproc_file
                    outf.write(line + "\n")
