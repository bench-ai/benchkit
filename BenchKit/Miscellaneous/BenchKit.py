import json
from pathlib import Path
from BenchKit.NeuralNetworks.Helpers import create_model_dir
from BenchKit.Train.Helpers import write_script
from .Settings import set_config, get_config
from .Verbose import verbose_logo
import argparse
import getpass
from .User import AuthenticatedUser, Credential, get_user_project, get_dataset_list, get_versions


def create_dataset():
    from BenchKit.Data.Helpers import create_dataset_dir
    create_dataset_dir()



def set_settings():
    with open("Config.json", "r") as file:
        x = json.load(file)

    save_path = Path(__file__).resolve().parent / "Config.json"
    with open(save_path, "w") as file:
        json.dump(x, file, indent=4)


def login():
    try:
        AuthenticatedUser.login()
    except Credential:
        print("Login Failed invalid credentials, you can also attempt to login manually using -inm or --loginm flag")


def logout():
    AuthenticatedUser.logout()
    write_config_template(lgn=False)
    set_settings()


def login_manual():
    username = input("Username: ")
    password = getpass.getpass()

    cred_dict = {
        "user_credentials": {
            "username": username,
            "password": password
        }
    }

    set_config(cred_dict)
    login()
    write_config()


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


def set_project(project_name: str):
    data = get_user_project(project_name)
    set_config({"project": data})
    write_config()


def update_dataset_config():
    config = get_config()
    project_id = config["project"]["id"]

    ds_list: list = get_dataset_list(project_id)

    set_config({"datasets": ds_list})
    write_config()


def update_code_version_config():
    version_list = get_versions()
    set_config({"code_versions": version_list})
    write_config()


def print_version():
    verbose_logo("V.0.0.24 ALPHA")


def load_project(project_name: str):
    write_config_template()
    set_project(project_name)
    update_dataset_config()
    update_code_version_config()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("action",
                        choices=["startproject", "logout", "setsettings"],
                        nargs="?",
                        default=None)

    parser.add_argument("input_value",
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

    if args.action == "startproject":

        if not args.input_value:
            raise ValueError("Project Name not provided")

        load_project(args.input_value)
        write_manager()
        create_dataset()
        create_model_dir()
        write_script()


if __name__ == '__main__':
    main()
