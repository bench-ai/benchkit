import json
import os

from BenchKit.Miscellaneous.Settings import get_main_url
from .user import request_executor


def get_logs(page: int,
             instance_id: str):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "get",
                               "log",
                               "b-k")

    response = request_executor("get",
                                url=request_url,
                                params={"page": page,
                                        "instance_id": instance_id})

    return json.loads(response.content)


def get_experiments(page: int,
                    version=None,
                    state=None):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "get",
                               "experiment")

    response = request_executor("get",
                                url=request_url,
                                params={"page": page,
                                        "version": version,
                                        "state": state})

    return json.loads(response.content)


def get_all_configs(instance_id: str):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "init",
                               "config")

    next_page = 1
    config_list = []
    id_list = []

    while next_page:
        response = request_executor("get",
                                    url=request_url,
                                    params={"instance_id": instance_id,
                                            "page": next_page})

        response_dict = json.loads(response.content)

        config_list += [i["parameters"] for i in response_dict["config_list"]]
        id_list += [i["id"] for i in response_dict["config_list"]]

        next_page = response_dict["next_page"]

    return config_list, id_list


def init_config(params: dict,
                evaluation_criteria: str):

    instance_id = os.getenv("INSTANCE_ID")

    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "init",
                               "config")

    response = request_executor("post",
                                url=request_url,
                                json={"instance_id": instance_id,
                                      "parameters": params,
                                      "evaluation_criteria": evaluation_criteria})

    response_dict = json.loads(response.content)

    return response_dict["id"]


def kill_server():
    request_url = os.path.join(get_main_url(), "api", "tracking", "server", "auto", "stop")

    instance_id = os.getenv("INSTANCE_ID")

    response = request_executor("delete",
                                url=request_url,
                                params={"instance_id": instance_id})

    return json.loads(response.content)
