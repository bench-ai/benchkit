import json
import os

from .user import request_executor
from benchkit.misc.settings import get_main_url


def get_logs(page: int, instance_id: str):
    request_url = os.path.join(get_main_url(), "api", "tracking", "get", "log", "b-k")

    response = request_executor(
        "get", url=request_url, params={"page": page, "instance_id": instance_id}
    )

    return json.loads(response.content)


def get_experiments(page: int, version=None, state=None):
    request_url = os.path.join(get_main_url(), "api", "tracking", "get", "experiment")

    response = request_executor(
        "get",
        url=request_url,
        params={"page": page, "version": version, "state": state},
    )

    return json.loads(response.content)


def get_all_configs(
    evaluation_criteria: str,
    sort_by: str,
    page: int,
    ascending: bool,
    running: bool | None = None,
    server_id: str | None = None,
) -> tuple[dict, str | None]:
    """
    Gets all configs in a paginated request

    :param sort_by: (str options) ["update_time", "criteria", "creation_time"]
    :param evaluation_criteria: (str)
    :param ascending: (bool)
    :param running: (bool | None) If True shows only model runs from active servers, if false shows models from
            terminated servers, if None shows all servers
    :param server_id: (str | None)
    :param page: (int)

    :return: 200 response tracker_config_data and next_page data
        tracker_config_data
            {"creation_timestamp": str(timestamp),
             "id": str,
             "parameters: dict,
             "server" str,
             "evaluation_criteria" str,
             "model_state": list[dict],
             "model_save": dict,
             "experiment_name": str,
             "version": int}
    """

    request_url = os.path.join(
        get_main_url(), "api", "tracking", "bk", "all", "config", "data"
    )

    response = request_executor(
        "get",
        url=request_url,
        params={
            "server_id": server_id,
            "evaluation_criteria": evaluation_criteria,
            "sort_by": sort_by,
            "ascending": ascending,
            "running": running,
            "page": page,
        },
    )

    response_dict = json.loads(response.content)

    return response_dict["tracker_config_data"], response_dict["next_page"]


def init_config(params: dict, evaluation_criteria: str) -> str:
    """

    :param params: The hyperparameters you wish to track
    :param evaluation_criteria: The evaluation criteria the model is judged on
    :return: the id of the config
    """

    instance_id = os.getenv("INSTANCE_ID")

    request_url = os.path.join(
        get_main_url(), "api", "tracking", "bk", "init", "config"
    )

    response = request_executor(
        "post",
        url=request_url,
        json={
            "instance_id": instance_id,
            "parameters": params,
            "evaluation_criteria": evaluation_criteria,
        },
    )

    response_dict = json.loads(response.content)

    return response_dict["id"]


def kill_server():
    request_url = os.path.join(
        get_main_url(), "api", "tracking", "server", "auto", "stop"
    )

    instance_id = os.getenv("INSTANCE_ID")

    response = request_executor(
        "delete", url=request_url, params={"instance_id": instance_id}
    )

    return json.loads(response.content)


def get_hyperparameters() -> dict:
    """
    Gets all hyperparmeters and evaluation criteria associated with this project

    :return: gets all hyperparameters and evaluation criterion associated with a project
    {"parameter_list": list(str),
     "evaluation_criteria_list: list(str),
     "project_id": str}
    """

    request_url = os.path.join(get_main_url(), "api", "tracking", "bk", "hyper-params")

    response = request_executor("get", url=request_url)

    return json.loads(response.content)
