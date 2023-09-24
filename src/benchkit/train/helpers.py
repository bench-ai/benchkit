import os.path
from pathlib import Path


def write_script():
    template_path = Path(__file__).resolve().parent / "train_script.txt"

    if not os.path.isfile("train_script.py"):
        with open(template_path) as read_file:
            with open("train_script.py", "w") as file:
                line = read_file.readline()
                while line:
                    file.write(line)
                    line = read_file.readline()
