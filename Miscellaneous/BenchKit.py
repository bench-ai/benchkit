import json
from pathlib import Path
from .Settings import set_config
from .Verbose import verbose_logo
import argparse
import getpass
from Data.Helpers import create_dataset_dir
from .User import AuthenticatedUser, Credential


def set_settings(settings_path: str):
    with open(settings_path, "r") as file:
        x = json.load(file)

    save_path = Path(__file__).resolve().parent / "Config.json"
    with open(save_path, "w") as file:
        json.dump(x, file)


def login():
    try:
        AuthenticatedUser.login()
    except Credential:
        print("Login Failed invalid credentials, you can also attempt to login manually using -inm or --loginm flag")


def logout():
    AuthenticatedUser.logout()

    cred_dict = {
        "user_credentials": {
            "username": "",
            "password": ""
        }
    }

    set_config(cred_dict)


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


def write_config():
    template_path = Path(__file__).resolve().parent / "configtemplate.txt"
    with open(template_path, "r") as f:
        with open("../Config.json", "w") as file:
            line = f.readline()
            while line:
                file.write(line)
                line = f.readline()


def write_manager():
    template_path = Path(__file__).resolve().parent / "manage.txt"
    with open(template_path, "r") as f:
        with open("manage.py", "w") as file:
            line = f.readline()
            while line:
                file.write(line)
                line = f.readline()


def start_dataset():
    create_dataset_dir()


def print_version():
    verbose_logo("V.0.0.1 ALPHA")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in",
                        "--login",
                        action='store_true',
                        required=False)

    parser.add_argument("-inm",
                        "--loginm",
                        action='store_true',
                        required=False)

    parser.add_argument("-ms",
                        "--migrate_settings",
                        required=False)

    parser.add_argument("-v",
                        "--version",
                        action='store_true',
                        required=False)

    parser.add_argument("-out",
                        "--logout",
                        action='store_true',
                        required=False)

    parser.add_argument("-ds",
                        "--dataset",
                        action='store_true',
                        required=False)

    parser.add_argument("-sp",
                        "--startproject",
                        action='store_true',
                        required=False)

    args = parser.parse_args()

    if args.login:
        login()

    if args.loginm:
        login_manual()

    if args.migrate_settings:
        set_settings(args.migrate_settings)

    if args.version:
        print_version()

    if args.logout:
        logout()

    if args.dataset:
        start_dataset()

    if args.startproject:
        write_config()
        write_manager()
        start_dataset()


if __name__ == '__main__':
    main()
