import os


def make_folder(path: str):
    """Creates a folder if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def count_lines(file_path: str) -> int:
    """Returns the number of lines from a hypothesis file"""
    with open(file_path, "rt") as f:
        return len(f.readlines())
