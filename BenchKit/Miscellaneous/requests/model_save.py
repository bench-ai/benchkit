import json
import os

from BenchKit.Miscellaneous.Settings import get_main_url
from .user import request_executor


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
