import json
import os
from functools import wraps
from pathlib import Path
import requests
from .Settings import get_main_url, convert_iso_time
from .Verbose import get_version
import torch


class AuthenticatedUser:
    cred_path = Path(__file__).resolve().parent / "credentials.json"

    def __init__(self):
        self.cred_dict = {}

        with open(self.cred_path, "r") as file:
            self.cred_dict.update(json.load(file))

    @property
    def api_key(self):
        return self.cred_dict["api_key"]

    @property
    def project_id(self):
        return self.cred_dict["project_id"]


def get_project_id() -> str:
    auth = AuthenticatedUser()
    return auth.project_id


def get_current_dataset(ds_name: str) -> dict:
    request_url = os.path.join(get_main_url(), "api", "dataset", "project", "list")
    response = request_executor("get",
                                url=request_url,
                                params={"page": 1,
                                        "name": ds_name})

    response = json.loads(response.content)

    return response["datasets"][0] if len(response["datasets"]) > 0 else None


def authorize_response(func):
    def run_request(*args, **kwargs) -> requests.Response:
        auth = AuthenticatedUser()
        header = {'project-id': auth.project_id,
                  'api-key': auth.api_key}

        kwargs.update({"headers": header})
        response: requests.Response = func(*args, **kwargs)
        return response

    @wraps(func)
    def wrapper(*args, **kwargs):

        response = run_request(*args, **kwargs)

        if response.status_code == 500:
            raise RuntimeError("500 Error server not working")

        elif str(response.status_code).startswith('2'):
            return response

        else:
            raise RuntimeError(f"Got error {response.status_code}, ERROR message: {json.loads(response.content)}")

    return wrapper


class Credential(Exception):
    pass


class UnknownRequest(Exception):
    pass


def test_login() -> bool:
    request_url = os.path.join(get_main_url(), "api", "auth", "project", "login")
    response = request_executor("get",
                                url=request_url)

    return json.loads(response.content)["success"]


def get_current_user() -> dict:
    request_url = os.path.join(get_main_url(), "api", "auth", "user", "current")
    response = request_executor("get",
                                url=request_url)

    return json.loads(response.content)


def create_dataset(dataset_name: str,
                   sample_count: int,
                   size: int):
    request_url = os.path.join(get_main_url(), "api", "dataset", "project", "list")

    response = request_executor("post",
                                url=request_url,
                                json={
                                    "name": dataset_name,
                                    "sample_count": sample_count,
                                    "size": size
                                })

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


def get_checkpoint_url(checkpoint_id):
    request_url = os.path.join(get_main_url(), "api", "tracking", "upload", "checkpoint")

    response = request_executor("get",
                                url=request_url,
                                params={
                                    "checkpoint_id": checkpoint_id
                                })

    return json.loads(response.content)


def update_server(instance_id: str,
                  progress: int | None = None,
                  current_step: int | None = None,
                  last_message: str | None = None):
    request_url = os.path.join(get_main_url(), "api", "tracking", "update", "progress")

    data_dict = {"instance_id": instance_id}

    if progress:
        data_dict["progress"] = progress

    if current_step:
        data_dict["current_step"] = current_step

    if last_message:
        data_dict["last_message"] = last_message

    response = request_executor("patch",
                                url=request_url,
                                json=data_dict)

    return json.loads(response.content)


def get_dataset_list():
    request_url = os.path.join(get_main_url(), "api", "dataset", "project", "list")
    next_page = 1
    dataset_list = []

    while next_page:
        response = request_executor("get",
                                    url=request_url,
                                    params={"page": 1})

        response = json.loads(response.content)
        next_page = response["next_page"]
        dataset_list.extend(response["datasets"])

    dataset_list.sort(key=lambda x: convert_iso_time(x["update_timestamp"]))

    return dataset_list


def get_chunk_count(dataset_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "chunk", "count")

    response = request_executor("get",
                                url=request_url,
                                params={
                                    "dataset_id": dataset_id,
                                })

    num_dict = json.loads(response.content)

    return num_dict["latest_chunk_num"] if num_dict["latest_chunk_num"] else 0


def get_user_project() -> dict:
    request_url = os.path.join(get_main_url(), "api", "project", "unique")

    response = request_executor("get",
                                url=request_url)

    return json.loads(response.content)


def get_post_url(dataset_id: str,
                 file_size: int,
                 file_path: str,
                 file_count: int):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    response = request_executor("post",
                                url=request_url,
                                json={"dataset_id": dataset_id,
                                      "size": file_size,
                                      "file_key": file_path,
                                      "file_count": file_count})

    return response


def delete_dataset(dataset_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    response = request_executor("delete",
                                url=request_url,
                                params={"dataset_id": dataset_id})

    return response


def get_dataset(dataset_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "get", "keys")

    response = request_executor("get",
                                url=request_url,
                                params={"dataset_id": dataset_id})

    content = json.loads(response.content)
    data_list = []
    data_list.extend([os.path.split(i)[-1] for i in content["key_list"]])

    while content["last_key"]:
        response = request_executor("get",
                                    url=request_url,
                                    params={"dataset_id": dataset_id,
                                            "last_key": content["last_key"]})
        content = json.loads(response.content)
        data_list.extend([os.path.split(i)[-1] for i in content["key_list"]])

    return data_list


def get_get_url(chunk_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    instance_id = os.getenv("INSTANCE_ID")

    response = request_executor("get",
                                url=request_url,
                                params={"chunk_id": chunk_id,
                                        "instance_id": instance_id})

    return json.loads(response.content)


def get_ds_chunks(dataset_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "list", "chunk")

    instance_id = os.getenv("INSTANCE_ID")

    next_page = 1

    ret_list = []

    while next_page:
        response = request_executor("get",
                                    url=request_url,
                                    params={"dataset_id": dataset_id,
                                            "page": next_page,
                                            "instance_id": instance_id})

        response_dict = json.loads(response.content)

        next_page = response_dict["next_page"]

        ret_list += response_dict["chunk_list"]

    return ret_list


def kill_server():
    request_url = os.path.join(get_main_url(), "api", "tracking", "server", "auto", "stop")

    instance_id = os.getenv("INSTANCE_ID")

    response = request_executor("delete",
                                url=request_url,
                                params={"instance_id": instance_id})

    return json.loads(response.content)


def upload_project_code(dependency_tar_size: int,
                        dependency_name: str,
                        train_script_tar_size: int,
                        train_script_name: str,
                        model_tar_size: int,
                        model_name: str,
                        dataloader_tar_size: int,
                        dataloader_name: str,
                        version: int):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "project",
                               "upload",
                               "code")

    response = request_executor("post",
                                url=request_url,
                                json={"dependency_tar_size": dependency_tar_size,
                                      "dependency_name": dependency_name,
                                      "train_script_tar_size": train_script_tar_size,
                                      "train_script_name": train_script_name,
                                      "model_tar_size": model_tar_size,
                                      "model_name": model_name,
                                      "dataloader_tar_size": dataloader_tar_size,
                                      "dataloader_name": dataloader_name,
                                      "version": version,
                                      "framework": "PYT",
                                      "benchkit_version": get_version(),
                                      "framework_version": torch.__version__})

    return json.loads(response.content)


def pull_project_code(version: int):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "project",
                               "get",
                               "code")

    response = request_executor("get",
                                url=request_url,
                                params={"version": version})

    return json.loads(response.content)


def get_versions():
    request_url = os.path.join(get_main_url(),
                               "api",
                               "project",
                               "b-k",
                               "version")

    response = request_executor("get",
                                url=request_url)

    return json.loads(response.content)


def delete_version(version: int):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "project",
                               "b-k",
                               "version")

    request_executor("delete",
                     url=request_url,
                     params={"version": version})


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


def init_config(params: dict):
    instance_id = os.getenv("INSTANCE_ID")

    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "init",
                               "config")

    response = request_executor("post",
                                url=request_url,
                                json={"instance_id": instance_id,
                                      "parameters": params})

    response_dict = json.loads(response.content)

    return response_dict["id"]


def get_all_configs(instance_id: str):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "init",
                               "config")

    next_page = 1
    config_list = []
    while next_page:
        response = request_executor("get",
                                    url=request_url,
                                    params={"instance_id": instance_id,
                                            "page": next_page})

        response_dict = json.load(response.content)

        config_list += response_dict["config_list"]
        next_page = response_dict["next_page"]

    return config_list


def make_time_series_graph(config_id: str,
                           name: str,
                           line_names: list[str],
                           x_name: str,
                           y_name: str) -> str:
    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "time-series")

    response = request_executor("post",
                                url=request_url,
                                json={
                                    "config_id": config_id,
                                    "name": name,
                                    "descriptors": {
                                        "line_names": line_names,
                                        "x_name": x_name,
                                        "y_name": y_name
                                    }
                                })

    response_dict = json.loads(response.content)

    return response_dict["id"]


def plot_time_series_point(graph_id: str,
                           line_name: str,
                           x_value: int,
                           y_value: float):
    request_url = os.path.join(get_main_url(),
                               "api",
                               "tracking",
                               "time-series",
                               "plot")

    response = request_executor("post",
                                url=request_url,
                                json={
                                    "graph_id": graph_id,
                                    "data_point": {
                                        "line_name": line_name,
                                        "x_value": x_value,
                                        "y_value": y_value
                                    }
                                })
    response_dict = json.loads(response.content)

    return response_dict


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


@authorize_response
def request_executor(req_type: str, **kwargs):
    if req_type.lower() == "post":
        response = requests.post(**kwargs)
    elif req_type.lower() == "patch":
        response = requests.patch(**kwargs)
    elif req_type.lower() == "get":
        response = requests.get(**kwargs)
    elif req_type.lower() == "put":
        response = requests.put(**kwargs)
    elif req_type.lower() == "delete":
        response = requests.delete(**kwargs)
    else:
        raise UnknownRequest("Options are post, patch, get, put, delete")

    return response
