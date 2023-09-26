import json

from requests import Response

from .user import request_executor
from benchkit.misc.settings import convert_iso_time
from benchkit.misc.settings import get_main_url


def get_current_dataset(ds_name: str) -> dict | None:
    """
    gets information on the dataset if it is present

    :param ds_name: the dataset name
    :return: a dictionary containing dataset info, or None

    The dictionary contains the following keys:
    - 'id': (str) UUID of the dataset
    - 'project': (str) The Projects UUID
    - 'size': (int) Size of the datasets in bytes
    - 'name': (str) The name of the dataset
    - 'sample_count': (int) the amount of samples in the dataset
    - 'creation_timestamp': (str) when the dataset was created in ISO
    - 'update_timestamp': (str) when the dataset was updated in ISO
    """

    request_url = "/".join([get_main_url(), "api", "dataset", "project", "list"])
    response = request_executor(
        "get", url=request_url, params={"page": 1, "name": ds_name}
    )

    response = json.loads(response.content)

    return response["datasets"][0] if len(response["datasets"]) > 0 else None


def create_dataset(dataset_name: str, sample_count: int, size: int) -> dict:
    """
    allocates space to upload a dataset to Bench AI

    :param dataset_name: the name of the dataset
    :param sample_count: how many samples the dataset has
    :param size: how many bytes the dataset takes up
    :return: dict

    The dictionary contains the following keys:
    - 'id': (str) UUID of the dataset
    - 'project': (str) The Projects UUID
    - 'size': (int) Size of the datasets in bytes
    - 'name': (str) The name of the dataset
    - 'sample_count': (int) the amount of samples in the dataset
    - 'creation_timestamp': (str) when the dataset was created in ISO
    - 'update_timestamp': (str) when the dataset was updated in ISO
    """
    request_url = "/".join([get_main_url(), "api", "dataset", "project", "list"])

    response = request_executor(
        "post",
        url=request_url,
        json={"name": dataset_name, "sample_count": sample_count, "size": size},
    )

    return json.loads(response.content)


def get_dataset_list() -> list[dict]:
    """
    gathers information on all datasets instantiated for this project
    :return: a list of dictionaries carrying metadata on the datasets

    The dictionary contains the following keys:
    - 'id': (str) UUID of the dataset
    - 'project': (str) The Projects UUID
    - 'size': (int) Size of the datasets in bytes
    - 'name': (str) The name of the dataset
    - 'sample_count': (int) the amount of samples in the dataset
    - 'creation_timestamp': (str) when the dataset was created in ISO
    - 'update_timestamp': (str) when the dataset was updated in ISO
    """
    request_url = "/".join([get_main_url(), "api", "dataset", "project", "list"])
    next_page = 1
    dataset_list = []

    while next_page:
        response = request_executor("get", url=request_url, params={"page": 1})

        response = json.loads(response.content)
        next_page = response["next_page"]
        dataset_list.extend(response["datasets"])

    dataset_list.sort(key=lambda x: convert_iso_time(x["update_timestamp"]))

    return dataset_list


def get_chunk_count(dataset_id: str) -> int:
    """
    tells you how many chunks are in a dataset

    :param dataset_id: the id of the dataset
    :return: the amount of chunks
    """
    request_url = "/".join([get_main_url(), "api", "dataset", "chunk", "count"])

    response = request_executor(
        "get",
        url=request_url,
        params={
            "dataset_id": dataset_id,
        },
    )

    num_dict = json.loads(response.content)

    return num_dict["latest_chunk_num"] if num_dict["latest_chunk_num"] else 0


def get_post_url(
    dataset_id: str, file_size: int, file_path: str, file_count: int
) -> Response:
    """
    gives a presigned post URL, this URL should be used for uploading chunks

    :param dataset_id: the uuid of the dataset
    :param file_size: the size in bytes of the chunk gzip
    :param file_path: where the chunk gzip is located
    :param file_count: the current number of the chunk, used for ordering the chunks
    :return: response object containing a dictionary

    The dictionary contains the following keys:
    - 'url': (str) the post url to use when uploading
    - 'fields': (dict) a dictionary of field metadata, SHOULD NOT BE ALTERED
    """
    request_url = "/".join([get_main_url(), "api", "dataset", "upload"])

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


def delete_dataset(dataset_id: str) -> Response:
    """
    deletes a dataset from the project

    :param dataset_id: the UUID of the dataset you wish to delete
    :return: A Response object containing a dict with the deleted datasets metadata

    The dictionary contains the following keys:
    - 'id': (str) UUID of the dataset
    - 'project': (str) The Projects UUID
    - 'size': (int) Size of the datasets in bytes
    - 'name': (str) The name of the dataset
    - 'sample_count': (int) the amount of samples in the dataset
    - 'creation_timestamp': (str) when the dataset was created in ISO
    - 'update_timestamp': (str) when the dataset was updated in ISO
    """
    request_url = "/".join([get_main_url(), "api", "dataset", "upload"])

    response = request_executor(
        "delete", url=request_url, params={"dataset_id": dataset_id}
    )

    return response


def get_get_url(chunk_id: str) -> str:
    """
    returns a presigned get url that acquires the chunk gzip

    :param chunk_id: the UUID of the dataset chunk
    :return: the get url
    """
    request_url = "/".join([get_main_url(), "api", "dataset", "upload"])

    response = request_executor("get", url=request_url, params={"chunk_id": chunk_id})

    return json.loads(response.content)


def get_ds_chunks(dataset_id: str) -> list[dict]:
    """
    gathers all the chunks belonging to a dataset

    :param dataset_id: the UUID of the dataset
    :return: a list of dicts, with each dict containing metadata info on the chunk

    The dictionary contains the following keys:
    - 'id': (str) UUID of the chunk
    - 'number': (int) the order number of the chunk
    - 'size': (int) Size of the chunk in bytes
    - 'file_count': (int) number of files present in the chunk
    - 'location': (str) where teh chunk is saved on the cloud
    - 'dataset_id': (str) UUID of the dataset the chunk belongs too
    """
    request_url = "/".join([get_main_url(), "api", "dataset", "list", "chunk"])

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
