import argparse
import sys
import os
from mosestokenizer import *

# from sacremoses import MosesDetokenizer as mdetok


def detokenize(infile: str, outfile: str):
    if infile == outfile:
        print("infile is equal to outfile")
        print(infile)
        print(outfile)
        exit()

    # check if infile exists
    if not os.path.exists(infile):
        print(f"File does not exist: {infile}")
        return

    # md = mdetok(lang='en')
    with MosesDetokenizer("en") as detokenize:
        # a = ['couldn', "'t", 'could', "n't"]
        # b = detokenize(a)

        with open(infile, "r") as inf, open(outfile, "wt") as outf:
            for line in inf:
                if " n't" in line:  # Moses detokenizer doesn't handle contractions
                    line = line.replace(" n't", "n't")

                tokens = line.strip().split(" ")

                text = detokenize(tokens)

                outf.write(text + "\n")


def _detokenize(path: str, filename: str, ext: str, tag: str = "detok"):
    infile = os.path.join(path, f"{filename}.{ext}")
    outfile = os.path.join(path, f"{filename}.{tag}.{ext}")
    detokenize(infile, outfile)


def detokenize_conll(path: str, folder="conll2014"):
    print("detokenizing conll2014")

    path = os.path.join(path, folder)

    _detokenize(
        path=os.path.join(path, "orig"),
        filename="conll2014.test",
        ext="orig",
        tag="detok",
    )
    _detokenize(
        path=os.path.join(path, "cor"),
        filename="conll2014.0.test",
        ext="cor",
        tag="detok",
    )
    _detokenize(
        path=os.path.join(path, "cor"),
        filename="conll2014.1.test",
        ext="cor",
        tag="detok",
    )


def detokenize_fce(path: str, folder="fce"):
    print("detokenizing fce")

    path = os.path.join(path, folder)

    filenames = [
        "fce.dev",
        "fce.test",
        "fce.train",
    ]
    extensions = [
        "orig",
        "cor",
    ]

    for filename in filenames:
        for ext in extensions:
            _detokenize(
                path=os.path.join(path, ext), filename=filename, ext=ext, tag="detok"
            )


def detoknize_wandi(path: str, folder="wi+locness"):
    print("detokenizing wi+locness")

    path = os.path.join(path, folder)

    filenames = [
        "ABCN.dev",
        "ABCN.test",
        "ABC.train",
        "A.dev",
        "B.dev",
        "C.dev",
        "N.dev",
    ]
    extensions = [
        "orig",
        "cor",
    ]

    for filename in filenames:
        for ext in extensions:
            _detokenize(
                path=os.path.join(path, ext), filename=filename, ext=ext, tag="detok"
            )


def detokenize_jfleg(path: str, folder="jfleg"):
    print("detokenizing jfleg")

    path = os.path.join(path, folder)

    filenames = [
        "jfleg.dev",
        "jfleg.test",
    ]

    # for .orig files
    for filename in filenames:
        _detokenize(
            path=os.path.join(path, "orig"), filename=filename, ext="orig", tag="detok"
        )

    filenames = [
        "jfleg.dev.0",
        "jfleg.dev.1",
        "jfleg.dev.2",
        "jfleg.dev.3",
        "jfleg.test.0",
        "jfleg.test.1",
        "jfleg.test.2",
        "jfleg.test.3",
    ]

    for filename in filenames:
        _detokenize(
            path=os.path.join(path, "cor"), filename=filename, ext="cor", tag="detok"
        )


def detokenize_all(path):
    # detokenize_conll(path)
    # detokenize_fce(path)
    detoknize_wandi(path)
    # detokenize_jfleg(path)


if __name__ == "__main__":
    """
    Uses the moses tokenizer to detokenize a file.
    The infile should be a text file, with one sentence per line.
    This script creates a new file at the outfile path, and writes
        one sentence per line.

    For example,
    Original sentence:
        Additionally , you could n't believe , but the vocalist said " All
        the chairs you are sitting down were set from him , " pointing
        at me ! !

    Detokenized sentence:
        Additionally, you could n't believe, but the vocalist said "All
        the chairs you are sitting down were set from him," pointing at
          me!!

    Note that it missed the contraction: could n't --> couldn't.
    I think Moses is coded to handle contractions as: couldn 't -> couldn't
    """
    # load CORPORA path from environment variable
    corpora_path = os.getenv("CORPORA")
    detokenize_all(path=corpora_path)
