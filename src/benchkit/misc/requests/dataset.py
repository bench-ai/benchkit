import json
import os

from .user import request_executor
from benchkit.misc.settings import convert_iso_time
from benchkit.misc.settings import get_main_url


def get_current_dataset(ds_name: str) -> dict:
    request_url = os.path.join(get_main_url(), "api", "dataset", "project", "list")
    response = request_executor(
        "get", url=request_url, params={"page": 1, "name": ds_name}
    )

    response = json.loads(response.content)

    return response["datasets"][0] if len(response["datasets"]) > 0 else None


def create_dataset(dataset_name: str, sample_count: int, size: int):
    request_url = os.path.join(get_main_url(), "api", "dataset", "project", "list")

    response = request_executor(
        "post",
        url=request_url,
        json={"name": dataset_name, "sample_count": sample_count, "size": size},
    )

    return json.loads(response.content)


def get_dataset_list():
    request_url = os.path.join(get_main_url(), "api", "dataset", "project", "list")
    next_page = 1
    dataset_list = []

    while next_page:
        response = request_executor("get", url=request_url, params={"page": 1})

        response = json.loads(response.content)
        next_page = response["next_page"]
        dataset_list.extend(response["datasets"])

    dataset_list.sort(key=lambda x: convert_iso_time(x["update_timestamp"]))

    return dataset_list


def get_chunk_count(dataset_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "chunk", "count")

    response = request_executor(
        "get",
        url=request_url,
        params={
            "dataset_id": dataset_id,
        },
    )

    num_dict = json.loads(response.content)

    return num_dict["latest_chunk_num"] if num_dict["latest_chunk_num"] else 0


def get_post_url(dataset_id: str, file_size: int, file_path: str, file_count: int):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    response = request_executor(
        "post",
        url=request_url,
        json={
            "dataset_id": dataset_id,
            "size": file_size,
            "file_key": file_path,
            "file_count": file_count,
        },
    )

    return response


def delete_dataset(dataset_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    response = request_executor(
        "delete", url=request_url, params={"dataset_id": dataset_id}
    )

    return response


def get_get_url(chunk_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "upload")

    # instance_id = os.getenv("INSTANCE_ID")

    response = request_executor("get", url=request_url, params={"chunk_id": chunk_id})

    return json.loads(response.content)


def get_ds_chunks(dataset_id: str):
    request_url = os.path.join(get_main_url(), "api", "dataset", "list", "chunk")

    next_page = 1

    ret_list = []

    while next_page:
        response = request_executor(
            "get", url=request_url, params={"dataset_id": dataset_id, "page": next_page}
        )

        response_dict = json.loads(response.content)

        next_page = response_dict["next_page"]

        ret_list += response_dict["chunk_list"]

    return ret_list
