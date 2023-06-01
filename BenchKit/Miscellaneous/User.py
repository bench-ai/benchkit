import json
import os
from functools import wraps
import requests
from .Settings import get_main_url, get_config, convert_iso_time
from dotenv import load_dotenv


def authorize_response(func):
    def run_request(*args, **kwargs) -> requests.Response:
        auth = AuthenticatedUser()
        header = {'project-id': auth.project_id,
                  'api-key': auth.api_key}

        print(header)

        kwargs.update({"headers": header})
        response: requests.Response = func(*args, **kwargs)
        return response

    @wraps(func)
    def wrapper(*args, **kwargs):

        response = run_request(*args, **kwargs)

        if response.status_code == 500:
            raise RuntimeError("500 Error server not working")

        elif response.status_code % 2 == 0:
            return response

        else:
            raise RuntimeError(f"Got error {response.status_code}, ERROR message: {json.loads(response.content)}")

    return wrapper


class Credential(Exception):
    pass


class UnknownRequest(Exception):
    pass


def test_login() -> bool:
    print("here")
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
                   project_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "user", "list")

    try:
        response = request_executor("post",
                                    url=request_url,
                                    json={
                                        "name": dataset_name,
                                        "project": project_id,
                                        "raw": True})

        code = 200
    except Credential as e:
        code = int(str(e).split(":")[-1])

    if code == 409:
        response = request_executor("get",
                                    url=request_url,
                                    params={"page": 1,
                                            "name": dataset_name,
                                            "project": project_id,
                                            "raw": True})

        if response.status_code == 200:
            return json.loads(response.content)["datasets"][0]
        else:
            raise RuntimeError("Unable to find Dataset")
    elif code == 200:
        return json.loads(response.content)
    else:
        raise RuntimeError("Unable to create Dataset")


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


def get_checkpoint_url(checkpoint_name: str,
                       experiment_name: str,
                       version: int):
    request_url = os.path.join(get_main_url(), "api", "tracking", "upload", "checkpoint")

    response = request_executor("get",
                                url=request_url,
                                params={
                                    "checkpoint_name": checkpoint_name,
                                    "experiment_name": experiment_name,
                                    "version": version
                                })

    return json.loads(response.content)


def update_server(instance_id: str,
                  progress: int | None = None,
                  current_step: int | None = None,
                  last_message: str | None = None):
    request_url = os.path.join(get_main_url(), "api", "tracking", "server", "launch")

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
    request_url = os.path.join(get_main_url(), "api", "dataset", "project", "get")
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


def patch_dataset_list(dataset_id: str,
                       length: int):
    request_url = os.path.join(get_main_url(), "api", "dataset", "user", "list")

    response = request_executor("patch",
                                url=request_url,
                                json={
                                    "id": dataset_id,
                                    "sample_count": length
                                })

    if response.status_code != 200:
        raise RuntimeError("Project does not exists please register it on bench")


def get_user_project() -> dict:
    request_url = os.path.join(get_main_url(), "api", "project", "unique")

    response = request_executor("get",
                                url=request_url)

    return json.loads(response.content)


def get_post_url(dataset_id: str,
                 file_size: int,
                 file_path: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    response = request_executor("post",
                                url=request_url,
                                json={"dataset_id": dataset_id,
                                      "file_size": file_size,
                                      "file_key": file_path})

    return response


def delete_dataset(dataset_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    response = request_executor("delete",
                                url=request_url,
                                json={"dataset_id": dataset_id})

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


def get_get_url(dataset_id: str,
                file_path: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    response = request_executor("get",
                                url=request_url,
                                params={"dataset_id": dataset_id,
                                        "file_key": file_path})

    return json.loads(response.content)


def get_gpu_count():
    request_url = os.path.join(get_main_url(), "api", "pricing", "plan", "train", "gpu-count")

    project_name = get_config()["project"]["name"]

    response = request_executor("get",
                                url=request_url,
                                params={"project_name": project_name})

    return json.loads(response.content)


def project_image_upload_url(tar_size: int,
                             version: int,
                             tar_name: int):
    request_url = os.path.join(get_main_url(), "api", "project", "image")

    project_name = get_config()["project"]["name"]

    response = request_executor("post",
                                url=request_url,
                                json={"project_name": project_name,
                                      "tarball_size": tar_size,
                                      "version": version,
                                      "tar_name": tar_name})

    return json.loads(response.content)


def get_versions():
    request_url = os.path.join(get_main_url(), "api", "project", "unique", "all", "images")

    response = request_executor("get",
                                url=request_url)

    return json.loads(response.content)


def delete_all_images():
    request_url = os.path.join(get_main_url(), "api", "project", "delete", "all", "images")

    project_name = get_config()["project"]["name"]

    request_executor("delete",
                     url=request_url,
                     json={"project_name": project_name})


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


class AuthenticatedUser:
    @property
    def api_key(self):
        load_dotenv()
        return os.getenv("API_KEY")

    @property
    def project_id(self):
        load_dotenv()
        return os.getenv("PROJECT_ID")
