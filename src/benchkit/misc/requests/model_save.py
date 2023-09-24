import json
import os

from .user import request_executor
from benchkit.misc.settings import get_main_url


def post_model_save_presigned_url(
    config_id: str, size_bytes: int, evaluation_criteria_value: float
) -> dict:
    """
    POST method, gets the user a presigned url to upload their model save
    :param config_id: The tracker `Config` id
    :type config_id: str
    :param evaluation_criteria_value: value of the criteria specified in `config`
    :type evaluation_criteria_value: float
    :param size_bytes:
    :type size_bytes: int
    :return: a dictionary with the presigned post url
    :rtype: dict
        presigned_post_url_dict
                "url": str
                "fields" dict
    """

    request_url = os.path.join(
        get_main_url(), "api", "tracking", "bk", "model", "save", "file"
    )

    response = request_executor(
        "post",
        url=request_url,
        json={
            "tracker_config_id": config_id,
            "evaluation_criteria_value": evaluation_criteria_value,
            "size_bytes": size_bytes,
        },
    )

    return json.loads(response.content)


def post_model_state_presigned_url(
    config_id: str, iteration: int, evaluation_criteria_value: float, size_bytes: int
) -> dict:
    """
    POST method, gets the user a presigned url to upload their model state
    :param config_id: The tracker `Config` id
    :type config_id: str
    :param iteration: The state iteration (Epoch, batch #, ...)
    :type iteration: int
    :param evaluation_criteria_value: value of the criteria specified in `config`
    :type evaluation_criteria_value: float
    :param size_bytes:
    :type size_bytes: int
    :return: a dictionary with the presigned post url
    :rtype: dict
        presigned_post_url_dict
                "url": str
                "fields" dict
    """

    request_url = os.path.join(
        get_main_url(), "api", "tracking", "bk", "model", "state", "file"
    )

    response = request_executor(
        "post",
        url=request_url,
        json={
            "tracker_config_id": config_id,
            "iteration": iteration,
            "evaluation_criteria_value": evaluation_criteria_value,
            "size_bytes": size_bytes,
        },
    )

    return json.loads(response.content)


def get_model_state_presigned_url(model_state_id: str) -> dict:
    """
    A get request that returns an url to the model state the user wishes to get

    :param model_state_id:
    :return: dict
        {url: (str)}
    """

    request_url = os.path.join(
        get_main_url(), "api", "tracking", "bk", "model", "state", "file"
    )

    response = request_executor(
        "get", url=request_url, params={"model_state_id": model_state_id}
    )

    return json.loads(response.content)


def get_model_save_presigned_url(model_save_id: str) -> dict:
    """
    A get request that returns an url to the model state the user wishes to get

    :param model_save_id:
    :return: dict
        {url: (str)}
    """

    request_url = os.path.join(
        get_main_url(), "api", "tracking", "bk", "model", "save", "file"
    )

    response = request_executor(
        "get", url=request_url, params={"model_save_id": model_save_id}
    )

    return json.loads(response.content)
