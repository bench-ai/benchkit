import json
import os
import tarfile
from pathlib import Path
import gzip
import shutil
import requests
from BenchKit.NeuralNetworks.Helpers import create_model_dir
from BenchKit.Train.Helpers import write_script
from .Settings import set_config, get_config
from .Verbose import verbose_logo
import argparse
from .User import get_user_project, get_dataset_list, get_versions, get_checkpoint_url, test_login


def create_dataset():
    from BenchKit.Data.Helpers import create_dataset_dir
    create_dataset_dir()


def set_settings():
    with open("Config.json", "r") as file:
        x = json.load(file)

    save_path = Path(__file__).resolve().parent / "Config.json"
    with open(save_path, "w") as file:
        json.dump(x, file, indent=4)


def logout():
    write_config_template(lgn=False)
    set_settings()
    if os.path.exists(".env"):
        os.remove(".env")


def login_manual():
    project_id = input("project-id: ")
    api_key = input("api-key: ")

    with open(".env", "w") as file:
        file.writelines([f"API_KEY={api_key}\n",
                         f"PROJECT_ID={project_id}\n"])

    try:
        test_login()
        write_config()
    except RuntimeError:
        logout()
        raise ValueError("Credentials invalid")


def write_config_template(lgn=True):
    template_path = Path(__file__).resolve().parent / "configtemplate.txt"
    with open(template_path, "r") as f:
        with open("Config.json", "w") as file:
            line = f.readline()
            while line:
                file.write(line)
                line = f.readline()

    set_settings()
    if lgn:
        login_manual()


def write_config():
    cfg = Path(__file__).resolve().parent / "Config.json"
    with open(cfg, "r") as f:
        cfg = json.load(f)
        with open("Config.json", "w") as file:
            json.dump(cfg, file, indent=4)


def write_manager():
    template_path = Path(__file__).resolve().parent / "manage.txt"
    with open(template_path, "r") as f:
        with open("manage.py", "w") as file:
            line = f.readline()
            while line:
                file.write(line)
                line = f.readline()


def set_project():
    data = get_user_project()
    set_config({"project": data})
    write_config()


def update_dataset_config():
    ds_list: list = get_dataset_list()
    set_config({"datasets": ds_list})
    write_config()


def update_code_version_config():
    version_list = get_versions()
    set_config({"code_versions": version_list})
    write_config()


def print_version():
    verbose_logo("V.0.0.28 ALPHA")


def load_project():
    write_config_template()
    set_project()
    update_dataset_config()
    update_code_version_config()


def get_checkpoint(checkpoint_name: str,
                   version: int,
                   experiment_name: str):

    request = get_checkpoint_url(f"{checkpoint_name}.tar.gz",
                                 experiment_name,
                                 version)

    mem_zip = requests.get(request)

    with open(f"{checkpoint_name}.tar.gz", 'wb') as f:
        f.write(mem_zip.content)

    with gzip.open(f"{checkpoint_name}.tar.gz", 'rb') as f_in:
        with open(f"{checkpoint_name}.tar", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with tarfile.open(f"{checkpoint_name}.tar", 'r') as tar:
        tar.extractall()

    os.remove(f"{checkpoint_name}.tar.gz")
    os.remove(f"{checkpoint_name}.tar")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("action",
                        choices=["startproject", "logout", "setsettings", "getcheckpoint"],
                        nargs="?",
                        default=None)

    parser.add_argument("input_value",
                        nargs='?',
                        default=None)

    parser.add_argument("input_value1",
                        nargs='?',
                        default=None)

    parser.add_argument("input_value2",
                        nargs='?',
                        default=None)

    parser.add_argument("-v",
                        "--version",
                        action='store_true',
                        required=False)

    args = parser.parse_args()

    if args.version:
        print_version()

    if args.action == "logout":
        logout()

    if args.action == "setsettings":
        set_settings()

    if args.action == "getcheckpoint":

        if not args.input_value:
            raise ValueError("Experiment name not provided")

        if not args.input_value1:
            raise ValueError("Checkpoint name not provided")

        if not args.input_value2:
            raise ValueError("Experiment name not provided")

        get_checkpoint(args.input_value1,
                       args.input_value2,
                       args.input_value)

    if args.action == "startproject":

        load_project()
        write_manager()
        create_dataset()
        create_model_dir()
        write_script()


if __name__ == '__main__':
    main()
