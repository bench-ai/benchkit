import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tabulate import tabulate
from tqdm import tqdm

from .settings import convert_timestamp
from .verbose import get_version
from .verbose import verbose_logo
from benchkit.misc.cli.server import display_tracker_config_plots
from benchkit.misc.cli.server import download_model_save
from benchkit.misc.cli.server import ModelRuns
from benchkit.misc.cli.server import ModelState
from benchkit.misc.requests.dataset import delete_dataset
from benchkit.misc.requests.dataset import get_dataset_list
from benchkit.misc.requests.server import get_experiments
from benchkit.misc.requests.server import get_logs
from benchkit.misc.requests.server import kill_server
from benchkit.misc.requests.user import get_user_project
from benchkit.misc.requests.user import test_login
from benchkit.misc.requests.version import delete_version
from benchkit.misc.requests.version import get_versions
from benchkit.misc.requests.version import pull_project_code
from benchkit.misc.utils.tar import extract_tar
from benchkit.nn.helpers import create_model_dir
from benchkit.train.helpers import write_script


def create_dataset():
    from benchkit.data.helpers import create_dataset_dir

    create_dataset_dir()


def logout():
    cred_path = Path(__file__).resolve().parent / "credentials.json"
    if os.path.exists(cred_path):
        os.remove(cred_path)


def login_manual(project_id: str, api_key: str):
    cred_path = Path(__file__).resolve().parent / "credentials.json"

    with open(cred_path, "w") as file:
        json.dump({"project_id": project_id, "api_key": api_key}, file)

    try:
        test_login()
    except RuntimeError as exc:
        logout()
        raise ValueError("Credentials invalid") from exc


def write_manager():
    template_path = Path(__file__).resolve().parent / "manage.txt"
    with open(template_path) as f:
        with open("manage.py", "w") as file:
            line = f.readline()
            while line:
                file.write(line)
                line = f.readline()


def write_dependency():
    if not os.path.isfile("dependencies.txt"):
        with open("dependencies.txt", "w"):
            pass


def print_version():
    verbose_logo(get_version())


def load_project(project_id: str, api_key: str):
    login_manual(project_id, api_key)


def gracefully_stop_server():
    kill_server()


def show_versions():
    version_dict = get_versions()
    if not version_dict:
        raise ValueError("No versions have been uploaded")

    df = pd.DataFrame(data=version_dict)
    df = df.drop(columns=["project_id"])
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

    return df


def del_versions():
    show_versions()
    version_number = int(input("Enter the number of the version you wish to delete: "))
    delete_version(version_number)


def show_project():
    df = pd.DataFrame(data=[get_user_project()])
    df = df.drop(columns=["project_folder"])
    print(tabulate(df, headers="keys", tablefmt="psql"))


def show_datasets():
    dataset_dict = get_dataset_list()

    if not dataset_dict:
        raise ValueError("No datasets have been uploaded")

    df = pd.DataFrame(data=dataset_dict)

    df["creation_timestamp"] = df["creation_timestamp"].apply(convert_timestamp)
    df["update_timestamp"] = df["update_timestamp"].apply(convert_timestamp)

    id_col = df["id"].values

    df = df.drop(columns=["id", "project"])

    print(tabulate(df, headers="keys", tablefmt="psql"))

    return df, id_col


def del_datasets():
    _, id_col = show_datasets()
    dataset_number = int(input("Enter the number of the dataset you wish to delete: "))
    delete_dataset(id_col[dataset_number])


def pull_version(version: int):
    code_dict = pull_project_code(version)

    for item in tqdm(code_dict.items()):
        k, v = item
        # TODO: Add timeout here
        mem_zip = requests.get(v)  # noqa S113
        with open(f"{k}.tar.gz", "wb") as f:
            f.write(mem_zip.content)

        extract_tar(f"{k}.tar.gz", "")
        os.remove(f"{k}.tar.gz")


# TODO: Refactor this function if possible. Too complex.
def show_experiments(version=None, state=None):  # noqa C901
    ext = False
    page = 1
    page_dict = {1: 0}
    while not ext:
        experiment_dict = get_experiments(page, version, state)

        server_dict = experiment_dict["servers"]
        next_page = experiment_dict["next_page"]

        if not server_dict:
            val_err_str = (
                f"No Experiments have been run for version: {version if version else 1}"
            )

            if state:
                val_err_str += f" and state: {state}."
            else:
                val_err_str += "."

            raise ValueError(val_err_str)

        df = pd.DataFrame(data=server_dict)

        df.index = df.index + page_dict[page]

        page_dict[next_page] = len(df) + page_dict[page]

        instance_series = df["instance_id"]

        df = df.drop(
            columns=["instance_id", "image", "killed_timestamp", "creation_timestamp"]
        )

        print(tabulate(df, headers="keys", tablefmt="psql", showindex=True))

        n_valid: bool = False
        p_valid: bool = False

        if next_page:
            print(f"The next page is {next_page}. ", end="")
            n_valid = True

        if page != 1:
            print(f"The previous page is {page - 1}", end="")
            p_valid = True

        print("\n")

        if n_valid:
            print("- Type 'n' to move to the next page.")

        if p_valid:
            print("- Type 'p' to move to the previous page.")

        print("- To see a server's logs, type '<server #> l'")
        print("- To see a server's Tracker Config, type '<server #> t'")

        str_inp = input("Enter: ")

        if str_inp.lower().startswith("n") and n_valid:
            page += 1
        elif str_inp.lower().startswith("p") and p_valid:
            page -= 1
        else:
            server_num, char = str_inp.split(" ")
            num = int(server_num)

            try:
                if char == "l":
                    show_logs(instance_series[num])
                elif char == "t":
                    show_model_runs(
                        evaluation_criteria=0,
                        sort_by="update_time",
                        ascending=True,
                        running=None,
                        server_id=instance_series[num],
                    )

            except (TypeError, ValueError):
                pass

            ext = True


# TODO: Refactor this function if possible. Too complex.
# flake8: noqa: C901
def show_model_runs(
    evaluation_criteria: str | int,
    sort_by: str,
    ascending: bool,
    running: bool | None = None,
    server_id: str | None = None,
) -> None:
    """
    Shows all Model Runs in a tabular form

    :param evaluation_criteria: The criteria used to evaluate your model, defined in the tracker config
    :param sort_by: The field to sort the query by options ["update_time", "creation_time", "criteria"]
    :param ascending: Whether the query should be sorted in ascending or descending order
    :param running: If True shows running models, if False shows models on not running servers, None shows all
    :param server_id: If present only show runs in relation to this server
    """

    generator = ModelRuns.get_model_runs_table_contents(
        evaluation_criteria,
        sort_by,
        ascending=ascending,
        running=running,
        server_id=server_id,
    )

    infinite = True
    first_iter = True
    send_flag = False

    (
        rows,
        model_state_list,
        model_save_list,
        server_id_list,
        tracker_config_id_list,
        next_page,
        headers,
    ) = next(generator)

    count = 0

    while infinite:
        print_list = [
            " - To see a TrackerConfig's states, type '<config #> s'\n",
            "- To see a TrackerConfig's save, type '<config #> v'\n",
            "- To see a TrackerConfig's metrics, type '<config #> m'\n",
        ]

        if next_page != 2 and (not first_iter):
            print_list.append("- Type 'p' to go to the previous page.\n")

        if not first_iter:
            (
                rows,
                model_state_list,
                model_save_list,
                server_id_list,
                tracker_config_id_list,
                next_page,
                headers,
            ) = generator.send(send_flag)

            if send_flag:
                count += len(tracker_config_id_list)
            else:
                count -= len(tracker_config_id_list)
        else:
            first_iter = False

        if next_page is not None:
            print_list.append("- Type 'n' to go to the next page.\n")

        index = list(range(count, len(tracker_config_id_list) + count))

        df = pd.DataFrame(rows, columns=headers, index=index)
        table = tabulate(df, headers="keys", tablefmt="psql", showindex=True)
        print(table)

        for i in print_list:
            print(i, end=" ")

        char = input("\nEnter: ")

        if char.startswith("p"):
            send_flag = False
        elif char.startswith("n"):
            send_flag = True
        elif len(char.split(" ")) == 2:
            config_num, mode = char.split(" ")
            config_num = int(config_num)
            if (config_num > max(index)) or (config_num < min(index)):
                raise IndexError("Index out of bounds")

            match mode:
                case "s":
                    if len(model_state_list[config_num - count]) > 1:
                        ModelState.show_and_download_states(
                            model_state_list[config_num - count],
                            rows[config_num - count]["experiment_name"],
                        )
                    else:
                        print("No model states are present")

                case "v":
                    if model_save_list[config_num - count]:
                        download_model_save(
                            model_save_list[config_num - count]["id"],
                            rows[config_num - count]["experiment_name"],
                        )
                    else:
                        print("No model saves are present")
                case "m":
                    display_tracker_config_plots(
                        tracker_config_id_list[config_num - count]
                    )
                case _:
                    raise ValueError("Valid modes are only s v m")

            infinite = False
        else:
            infinite = False


def show_logs(instance_id: str):
    ext = False
    page = 1
    current_timestamp = ""
    file_line = 0

    while not ext:
        log_dict = get_logs(page=page, instance_id=instance_id)

        if current_timestamp == log_dict["update_timestamp"]:
            if log_dict["next_page"]:
                page = log_dict["next_page"]
                file_line = 0
            else:
                if not log_dict["state"] == "SRU":
                    ext = True
                else:
                    time.sleep(5)
        else:
            current_timestamp = log_dict["update_timestamp"]

            # TODO: Consider adding a timeout here.
            mem_file = requests.get(log_dict["log_url"])  # noqa S113
            content = mem_file.text
            lines = content.splitlines()

            for idx, line in enumerate(lines):
                if idx == file_line:
                    print(line)
                    file_line += 1


def main():  # noqa C901
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        choices=[
            "start-project",
            "logout",
            "show-ds",
            "del-ds",
            "project-info",
            "show-vs",
            "del-vs",
            "pull-vs",
            "stop-svr",
            "show-ex",
            "show-runs",
        ],
        nargs="?",
        default=None,
    )

    parser.add_argument("input_value", nargs="?", default=None)

    parser.add_argument("input_value1", nargs="?", default=None)

    parser.add_argument("input_value2", nargs="?", default=None)

    parser.add_argument("-v", "--version", action="store_true", required=False)

    parser.add_argument("-s", "--state", type=str, required=False)

    parser.add_argument("-cv", "--code_version", type=int, required=False)

    parser.add_argument(
        "-sb",
        "--sort_by",
        default="update_time",
        choices=["update_time", "criteria", "creation_time"],
    )

    parser.add_argument(
        "-run", "--running", choices=["None", "True", "False"], default="None"
    )

    parser.add_argument("-asc", "--ascending", action="store_true")

    parser.add_argument("-sid", "--server_id", type=str, required=False)

    args = parser.parse_args()

    if args.version and not args.action:
        print_version()

    if args.action == "logout":
        logout()

    if args.action == "show-ds":
        show_datasets()

    if args.action == "del-ds":
        del_datasets()

    if args.action == "project-info":
        show_project()

    if args.action == "del-vs":
        del_versions()

    if args.action == "show-vs":
        show_versions()

    if args.action == "stop-svr":
        gracefully_stop_server()

    if args.action == "pull-vs":
        if not args.input_value:
            raise ValueError("Project version was not provided")

        pull_version(args.input_value)

    if args.action == "start-project":
        if not args.input_value:
            raise ValueError("Project id was not provided")

        if not args.input_value1:
            raise ValueError("Apikey was not provided")

        load_project(args.input_value, args.input_value1)
        write_manager()
        create_dataset()
        create_model_dir()
        write_script()
        write_dependency()

    if args.action == "show-ex":
        show_experiments(args.code_version, args.state)

    if args.action == "show-runs":
        if not args.input_value:
            raise ValueError("Evaluation Criteria was not provided")
        else:
            eval_crit = args.input_value

        show_model_runs(
            eval_crit,
            args.sort_by,
            args.ascending,
            running=eval(args.running),
            server_id=args.server_id,
        )


if __name__ == "__main__":
    main()
