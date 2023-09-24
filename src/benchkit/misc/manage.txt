import argparse
import os

from benchkit.data.helpers import create_dataset_zips, test_dataloading, run_upload
from datasets.project_datasets import main
from benchkit.misc.utils.tar import generate_tar
from benchkit.misc.requests.version import upload_project_code
from benchkit.misc.utils.bucket import upload_using_presigned_url
from colorama import Fore, Style
from pathlib import Path
from tqdm import tqdm


def create_datasets():
    return main()


def migrate_code(version: int):
    print(Fore.RED + "Starting Zipping process" + Style.RESET_ALL)
    dep_path = Path(__file__).resolve().parent / "dependencies.txt"
    dep_tar = generate_tar("dependency", str(dep_path))

    mod_path = Path(__file__).resolve().parent / "models"
    mod_tar = generate_tar("models", str(mod_path))

    dat_path = Path(__file__).resolve().parent / "datasets"
    dat_tar = generate_tar("datasets", str(dat_path))

    ts_path = Path(__file__).resolve().parent / "train_script.py"
    ts_tar = generate_tar("train-script", str(ts_path))

    print(Fore.GREEN + "Completed Zipping process" + Style.RESET_ALL)

    data_dict: dict = upload_project_code(os.path.getsize(dep_tar),
                                          dep_tar,
                                          os.path.getsize(ts_tar),
                                          ts_tar,
                                          os.path.getsize(mod_tar),
                                          mod_tar,
                                          os.path.getsize(dat_tar),
                                          dat_tar,
                                          version)

    print(Fore.RED + "Starting Upload process" + Style.RESET_ALL)

    for set_dict in tqdm(data_dict.items(), colour="blue"):
        _, v = set_dict
        upload_using_presigned_url(v["url"],
                                   os.path.split(v["fields"]["key"])[-1],
                                   os.path.split(v["fields"]["key"])[-1],
                                   v["fields"])

    print(Fore.GREEN + "Completed Upload process" + Style.RESET_ALL)

    os.remove(dep_tar)
    os.remove(mod_tar)
    os.remove(dat_tar)
    os.remove(ts_tar)


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("action",
                        nargs='?',
                        choices=["migrate-data", "reset", "migrate-code"])

    parser.add_argument("input_value",
                        nargs='?',
                        default=None)

    parser.add_argument("--zip",
                        help="create zip files",
                        action="store_true",
                        required=False)

    parser.add_argument("--tdl",
                        help="test data loader",
                        action="store_true",
                        required=False)

    parser.add_argument("--up",
                        help="upload dataset",
                        action="store_true",
                        required=False)

    args = parser.parse_args()

    if args.action == "migrate-data":

        arg_list = []

        if args.input_value is not None:
            for list_arg in create_datasets():
                if list_arg[2] == args.input_value:
                    arg_list = [list_arg]
            if not arg_list:
                raise ValueError(f"No such dataset named {args.input_value}")
        else:
            arg_list = create_datasets()

        for input_args in arg_list:

            p_ds, c_ds, name = input_args

            argument_list = [args.zip, args.tdl, args.up]

            if argument_list == ([False] * len(argument_list)):
                args.zip, args.tdl, args.up = True, True, True

            if args.zip:
                create_dataset_zips(p_ds, name)

            if args.tdl:
                test_dataloading(name, c_ds)

            if args.up:
                run_upload(name)

    elif args.action == "migrate-code":

        print(Fore.RED + "Starting Code Migration" + Style.RESET_ALL)

        if not args.input_value:
            args.input_value = 1

        migrate_code(args.input_value)


if __name__ == '__main__':
    cli()
