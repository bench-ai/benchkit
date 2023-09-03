import os
import pprint

import pandas as pd

from BenchKit.Miscellaneous.requests.model_save import get_model_state_presigned_url, get_model_save_presigned_url
from BenchKit.Miscellaneous.requests.server import get_all_configs, get_hyperparameters
from tabulate import tabulate

from BenchKit.Miscellaneous.utils.tar import download_file
from BenchKit.tracking.visualizer import Plotter, get_graph_n_points, get_display_matrix
import matplotlib.pyplot as plt


def get_table_headers():
    hyperparameter_dict = get_hyperparameters()

    headers: list = hyperparameter_dict["parameter_list"]
    headers.extend([
        "update_timestamp",
        "creation_timestamp",
        "best_evaluation",
        "current_evaluation",
        "last_iteration",
        "experiment_name",
        "code_version",
    ])

    return headers, hyperparameter_dict["evaluation_criteria_list"]


def show_evaluation_criteria(verbose=False):
    _, evaluation_criteria_list = get_table_headers()
    evaluation_criteria_dict = {idx: name for idx, name in enumerate(evaluation_criteria_list)}

    if verbose:
        pprint.pprint(evaluation_criteria_dict)

    return evaluation_criteria_dict


def convert_tracker_config(tracker_row: dict) -> tuple[dict, list[dict], str, str, str]:
    """
    takes a tracker config dict and adds relevant data to ir such as best eval score, current eval score iteration etc
    :param tracker_row:
    :return:
        lists of all relevant values:
            rows (dict)
            model_state_list (list )
            model_save (dict)
            server_id (str)
            tracker_config_id (str)
    """

    if tracker_row.get("model_save"):
        best_evaluation = tracker_row["model_save"]["evaluation_criteria_value"]
    else:
        best_evaluation = None

    if len(tracker_row["model_state"]):
        current_model_state = max(tracker_row["model_state"], key=lambda x: x["iteration"])
        current_evaluation_criteria = current_model_state["evaluation_criteria_value"]
        current_iteration = current_model_state["iteration"]
    else:
        current_evaluation_criteria = None
        current_iteration = None

    tracker_row.pop("evaluation_criteria")
    tracker_config_id = tracker_row.pop("id")
    server_id = tracker_row.pop("server")
    model_save = tracker_row.pop("model_save")
    model_state_list = tracker_row.pop("model_state")
    experiment_name = tracker_row.pop("experiment_name")
    version = tracker_row.pop("version")

    row_dict = {**tracker_row["parameters"]}

    row_dict.update({
        "best_evaluation": best_evaluation,
        "current_evaluation": current_evaluation_criteria,
        "last_iteration": current_iteration,
        "experiment_name": experiment_name,
        "code_version": version
    })

    return row_dict, model_state_list, model_save, server_id, tracker_config_id


def display_table(evaluation_criteria: str | int,
                  sort_by: str,
                  ascending: bool,
                  running: bool | None = None,
                  server_id: str | None = None):
    headers, evaluation_criteria_list = get_table_headers()
    evaluation_criteria_dict = {idx: name for idx, name in enumerate(evaluation_criteria_list)}

    if isinstance(evaluation_criteria, int):
        criteria = evaluation_criteria_dict[evaluation_criteria]
    else:
        criteria = evaluation_criteria

    current_page = 1

    while True:

        tracker_config_data, next_page = get_all_configs(criteria,
                                                         sort_by,
                                                         current_page,
                                                         ascending,
                                                         running=running,
                                                         server_id=server_id)

        rows, model_state_list, model_save_list, server_id_list, tracker_config_id_list = list(
            zip(*list(map(convert_tracker_config, tracker_config_data))))

        forward = yield rows, model_state_list, model_save_list, server_id_list, tracker_config_id_list, next_page, headers

        if forward and (not next_page):
            raise ValueError("Cannot move forward, current page is the last page")
        elif not forward and (current_page == 1):
            raise ValueError("Cannot move Backward, current page is the first page")
        elif not forward:
            current_page -= 1
        elif forward:
            current_page += 1


def display_state_list(model_state_list: dict):
    df = pd.DataFrame.from_records(data=model_state_list)

    model_state_id_list = df["id"]
    df.drop(columns="id")

    table = tabulate(df, headers='keys', tablefmt='psql', showindex=True)
    print(table)
    return model_state_id_list


def download_model_state(model_state_id: str,
                         experiment_name: str):
    get_url: dict = get_model_state_presigned_url(model_state_id)

    model_state_save_location = f"./{experiment_name}/states/"

    os.makedirs(f"./{experiment_name}/states/", exist_ok=True)
    os.makedirs(f"./{experiment_name}/save/", exist_ok=True)

    download_file(model_state_save_location, get_url["url"])


def download_model_save(model_save_id: str, experiment_name: str):
    get_url: dict = get_model_save_presigned_url(model_save_id)
    model_save_location = f"./{experiment_name}/save/"
    os.makedirs(model_save_location, exist_ok=True)
    download_file(model_save_location, get_url["url"])


def show_and_download_checkpoints(model_state_list: dict, experiment_name):
    state_id_list = display_state_list(model_state_list)

    checkpoint_number = int(input("Enter the number of the checkpoint you wish to download: "))

    if checkpoint_number > (len(state_id_list) - 1):
        raise IndexError("Index out of bounds")

    download_model_state(state_id_list[checkpoint_number], experiment_name)


def display_tracker_config_plots(tracker_config_id: str):
    plotter = Plotter()

    graph_list = get_graph_n_points(tracker_config_id)

    rows, columns = get_display_matrix(len(graph_list))

    fig, axs = plt.subplots(rows, columns, squeeze=False)

    mx = max(rows, columns)

    for idx, (data_dict, graph_description) in enumerate(graph_list):
        x = idx // mx
        y = idx % mx

        plotter(graph_description["graph_type"],
                data_dict,
                graph_description,
                axs[x, y])

    plt.show()
