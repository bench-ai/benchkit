import json
import os

import torch

from .user import request_executor
from benchkit.misc.settings import get_main_url
from benchkit.misc.verbose import get_version


def delete_version(version: int):
    request_url = os.path.join(get_main_url(), "api", "project", "b-k", "version")

    request_executor("delete", url=request_url, params={"version": version})


def get_versions():
    request_url = os.path.join(get_main_url(), "api", "project", "b-k", "version")

    response = request_executor("get", url=request_url)

    return json.loads(response.content)


def pull_project_code(version: int):
    request_url = os.path.join(get_main_url(), "api", "project", "get", "code")

    response = request_executor("get", url=request_url, params={"version": version})

    return json.loads(response.content)


def upload_project_code(
    dependency_tar_size: int,
    dependency_name: str,
    train_script_tar_size: int,
    train_script_name: str,
    model_tar_size: int,
    model_name: str,
    dataloader_tar_size: int,
    dataloader_name: str,
    version: int,
):
    request_url = os.path.join(get_main_url(), "api", "project", "upload", "code")

    response = request_executor(
        "post",
        url=request_url,
        json={
            "dependency_tar_size": dependency_tar_size,
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
            "framework_version": torch.__version__,
        },
    )

    return json.loads(response.content)
