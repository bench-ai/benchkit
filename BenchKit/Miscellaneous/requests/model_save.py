import json
import os
from BenchKit.Miscellaneous.Settings import get_main_url
from .user import request_executor
from BenchKit.tracking.config import Config


def post_model_save_presigned_url(config: Config,
                                  size_bytes: int,
                                  evaluation_criteria_value: float) -> dict:
    """
        POST method, gets the user a presigned url to upload their model save
        :param config: The Tracker Config
        :type config: Config
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

    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "model",
                               "save",
                               "file")

    response = request_executor("post",
                                url=request_url,
                                json={
                                    "config": config.config_id,
                                    "evaluation_criteria_value": evaluation_criteria_value,
                                    "size_bytes": size_bytes
                                })

    return json.loads(response.content)


def post_model_state_presigned_url(config: Config,
                                   iteration: int,
                                   evaluation_criteria_value: float,
                                   size_bytes: int) -> dict:

    """
    POST method, gets the user a presigned url to upload their model state
    :param config: The Tracker Config
    :type config: Config
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

    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "model",
                               "state",
                               "file")

    response = request_executor("post",
                                url=request_url,
                                json={
                                    "config": config.config_id,
                                    "iteration": iteration,
                                    "evaluation_criteria_value": evaluation_criteria_value,
                                    "size_bytes": size_bytes
                                })

    return json.loads(response.content)


def get_checkpoint_url(checkpoint_id):
    request_url = os.path.join(get_main_url(), "api", "tracking", "upload", "checkpoint")

    response = request_executor("get",
                                url=request_url,
                                params={
                                    "checkpoint_id": checkpoint_id
                                })

    return json.loads(response.content)


def post_checkpoint_url(checkpoint_name: str):
    if not checkpoint_name.endswith(".tar.gz"):
        raise ValueError("Checkpoint must be a tar.gz")

    instance_id = os.getenv("INSTANCE_ID")
    request_url = os.path.join(get_main_url(), "api", "tracking", "upload", "checkpoint")

    response = request_executor("post",
                                url=request_url,
                                json={
                                    "checkpoint_name": checkpoint_name,
                                    "instance_id": instance_id
                                })

    return json.loads(response.content)


def delete_checkpoints(checkpoint_id: str):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "list",
                               "checkpoint")

    response = request_executor("delete",
                                url=request_url,
                                params={"checkpoint_id": checkpoint_id})

    return json.loads(response.content)


def list_all_checkpoints():
    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "list",
                               "checkpoint")

    response = request_executor("get",
                                url=request_url)

    return json.loads(response.content)
