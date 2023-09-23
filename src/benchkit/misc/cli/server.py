import os

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from benchkit.misc.requests.model_save import get_model_save_presigned_url
from benchkit.misc.requests.model_save import get_model_state_presigned_url
from benchkit.misc.requests.server import get_all_configs
from benchkit.misc.requests.server import get_hyperparameters
from benchkit.misc.utils.tar import download_file
from benchkit.tracking.visualizer import get_display_matrix
from benchkit.tracking.visualizer import get_graph_n_points
from benchkit.tracking.visualizer import Plotter


class ModelRuns:

    """
    Methods related to displaying and getting Model runs data
    """

    @staticmethod
    def _get_table_headers() -> tuple[list[str], list[str]]:
        """
        Gets hyperparameters associated with this project, uses these values as columns for the model runs tables,
        adds some basic fields to it as well.
        :return: the headers for the table, and evaluation criterion used in the project
        """
        hyperparameter_dict = get_hyperparameters()

        headers: list = hyperparameter_dict["parameter_list"]
        headers.extend(
            [
                "update_timestamp",
                "creation_timestamp",
                "best_evaluation",
                "current_evaluation",
                "last_iteration",
                "experiment_name",
                "code_version",
            ]
        )

        return headers, hyperparameter_dict["evaluation_criteria_list"]

    @staticmethod
    def _convert_tracker_config(
        tracker_row: dict,
    ) -> tuple[dict, list[dict], dict, str, str]:
        """
        takes a tracker config dict and adds relevant data to it such as best eval score, current eval score iteration etc
        :param tracker_row:
        :return:
            lists of all relevant values:
                rows (dict)
                model_state_list (list (dict))
                model_save (dict)
                server_id (str)
                tracker_config_id (str)
        """

        if tracker_row.get("model_save"):
            best_evaluation = tracker_row["model_save"]["evaluation_criteria_value"]
        else:
            best_evaluation = None

        if len(tracker_row["model_state"]):
            current_model_state = max(
                tracker_row["model_state"], key=lambda x: x["iteration"]
            )
            current_evaluation_criteria = current_model_state[
                "evaluation_criteria_value"
            ]
            current_iteration = current_model_state["iteration"]
            last_timestamp = current_model_state["creation_timestamp"]
        else:
            current_evaluation_criteria = None
            current_iteration = None
            last_timestamp = None

        tracker_row.pop("evaluation_criteria")
        tracker_config_id = tracker_row.pop("id")
        server_id = tracker_row.pop("server")
        model_save = tracker_row.pop("model_save")
        model_state_list = tracker_row.pop("model_state")
        experiment_name = tracker_row.pop("experiment_name")
        version = tracker_row.pop("version")

        row_dict = {**tracker_row["parameters"]}

        row_dict.update(
            {
                "best_evaluation": best_evaluation,
                "current_evaluation": current_evaluation_criteria,
                "last_iteration": current_iteration,
                "experiment_name": experiment_name,
                "code_version": version,
                "update_timestamp": last_timestamp,
                "creation_timestamp": tracker_row["creation_timestamp"],
            }
        )

        return row_dict, model_state_list, model_save, server_id, tracker_config_id

    @staticmethod
    def get_model_runs_table_contents(
        evaluation_criteria: str | int,
        sort_by: str,
        ascending: bool,
        running: bool | None = None,
        server_id: str | None = None,
    ):
        """
        A generator that does previous page and next page request, based on the value sent in forward variable

        :param evaluation_criteria: the name of the criterion or a number representing its index
        :param sort_by: The field to sort the query by options ["update_time", "creation_time", "criteria"]
        :param ascending: Whether the query should be sorted in ascending or descending order
        :param running: If True shows running models, if False shows models on not running servers, None shows all
        :param server_id: If present only show runs in relation to this server
        :return:
            - rows (list(dict)): The TrackerConfig filled with its corresponding hyperparameters and metrics
            - model_state_list (list(list(dict))): A list of all states corresponding to a tracker config
            - model_save_list (list(dict)): A list of all saves corresponding to a tracker config
            - next_page (str | None): Whether there is another page of data
            - headers (list(str)): All the table headers
        """

        headers, evaluation_criteria_list = ModelRuns._get_table_headers()
        evaluation_criteria_dict = {
            idx: name for idx, name in enumerate(evaluation_criteria_list)
        }

        if isinstance(evaluation_criteria, int):
            criteria = evaluation_criteria_dict[evaluation_criteria]
        else:
            criteria = evaluation_criteria

        current_page = 1

        while True:
            tracker_config_data, next_page = get_all_configs(
                criteria,
                sort_by,
                current_page,
                ascending,
                running=running,
                server_id=server_id,
            )

            (
                rows,
                model_state_list,
                model_save_list,
                server_id_list,
                tracker_config_id_list,
            ) = list(
                zip(
                    *list(map(ModelRuns._convert_tracker_config, tracker_config_data)),
                    strict=True,
                )
            )

            forward = yield (
                rows,
                model_state_list,
                model_save_list,
                server_id_list,
                tracker_config_id_list,
                next_page,
                headers,
            )

            if forward and (not next_page):
                raise ValueError("Cannot move forward, current page is the last page")
            elif not forward and (current_page == 1):
                raise ValueError("Cannot move Backward, current page is the first page")
            elif not forward:
                current_page -= 1
            elif forward:
                current_page += 1


class ModelState:
    """
    Methods related to displaying and getting Model states
    """

    @staticmethod
    def show_and_download_states(model_state_list: dict, experiment_name) -> None:
        """
        Displays all model states and allows users to download them
        :param model_state_list:
        :param experiment_name:
        :return:
        """
        state_id_list = ModelState._display_state_list(model_state_list)

        checkpoint_number = int(
            input("Enter the number of the checkpoint you wish to download: ")
        )

        if checkpoint_number > (len(state_id_list) - 1):
            raise IndexError("Index out of bounds")

        ModelState._download_model_state(
            state_id_list[checkpoint_number], experiment_name
        )

    @staticmethod
    def _display_state_list(model_state_list: dict) -> pd.Series:
        """
        Displays the model states, associated to a tracker config in a table
        :param model_state_list:
        :return: returns a series of a model state_id's
        """
        df = pd.DataFrame.from_records(data=model_state_list)

        model_state_id_list = df["id"]
        df.drop(columns="id")

        table = tabulate(df, headers="keys", tablefmt="psql", showindex=True)
        print(table)
        return model_state_id_list

    @staticmethod
    def _download_model_state(model_state_id: str, experiment_name: str) -> None:
        """
        Downloads Model state locally
        :param model_state_id:
        :param experiment_name:
        """
        get_url: dict = get_model_state_presigned_url(model_state_id)

        model_state_save_location = f"./{experiment_name}/states/"

        os.makedirs(f"./{experiment_name}/states/", exist_ok=True)
        os.makedirs(f"./{experiment_name}/save/", exist_ok=True)

        download_file(model_state_save_location, get_url["url"])


def download_model_save(model_save_id: str, experiment_name: str) -> None:
    """
    Downloads Model save locally
    :param model_save_id:
    :param experiment_name:
    """
    get_url: dict = get_model_save_presigned_url(model_save_id)
    model_save_location = f"./{experiment_name}/save/"
    os.makedirs(model_save_location, exist_ok=True)
    download_file(model_save_location, get_url["url"])


def display_tracker_config_plots(tracker_config_id: str) -> None:
    """
    Based on tracker config it wil display all associated metrics
    :param tracker_config_id:
    """
    plotter = Plotter()

    graph_list = get_graph_n_points(tracker_config_id)

    rows, columns = get_display_matrix(len(graph_list))

    fig, axs = plt.subplots(rows, columns, squeeze=False)

    mx = max(rows, columns)

    for idx, (data_dict, graph_description) in enumerate(graph_list):
        x = idx // mx
        y = idx % mx

        plotter(
            graph_description["graph_type"], data_dict, graph_description, axs[x, y]
        )

    plt.show()
