from colorama import Fore
from colorama import Style
from pathlib import Path


def get_version():
    return "0.0.64"


def verbose_logo(version: str):
    package_dir = Path(__file__).resolve().parent / "bench_ascii_logo.txt"

    with open(package_dir, "r") as file:

        new_line = ""
        status = False
        status_line = ""
        line_list = []
        for line in file.readlines():

            for char in line:
                if char != "#":
                    if not status:
                        new_line += char
                    else:
                        status = False
                        new_line += ("{}{}{}".format(Fore.YELLOW, status_line, Style.RESET_ALL) + char)
                        status_line = ""
                else:
                    status = True
                    status_line += char

            line_list.append(new_line)

    print(line_list[-1])
    print()
    print(Fore.GREEN + "   " + version + Style.RESET_ALL)
