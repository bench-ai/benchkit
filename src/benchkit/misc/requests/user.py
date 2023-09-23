import json
import os
from functools import wraps
from pathlib import Path

import requests

from benchkit.misc.settings import get_main_url


class AuthenticatedUser:
    cred_path = Path(__file__).resolve().parent.parent / "credentials.json"

    def __init__(self):
        self.cred_dict = {}

        with open(self.cred_path) as file:
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


def authorize_response(func):
    def run_request(*args, **kwargs) -> requests.Response:
        auth = AuthenticatedUser()
        header = {"project-id": auth.project_id, "api-key": auth.api_key}

        kwargs.update({"headers": header})
        response: requests.Response = func(*args, **kwargs)
        return response

    @wraps(func)
    def wrapper(*args, **kwargs):
        response = run_request(*args, **kwargs)

        if response.status_code == 500:
            raise RuntimeError("500 Error server not working")

        elif str(response.status_code).startswith("2"):
            return response

        else:
            raise RuntimeError(
                f"Got error {response.status_code}, ERROR message: {json.loads(response.content)}"
            )

    return wrapper


class UnknownRequestError(Exception):
    pass


def test_login() -> bool:
    request_url = os.path.join(get_main_url(), "api", "auth", "project", "login")
    response = request_executor("get", url=request_url)

    return json.loads(response.content)["success"]


def get_user_project() -> dict:
    request_url = os.path.join(get_main_url(), "api", "project", "unique")

    response = request_executor("get", url=request_url)

    return json.loads(response.content)


# flake8: noqa: S113
@authorize_response
def request_executor(req_type: str, **kwargs):
    # TODO: Consider adding a timeout and catching the errors.
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
        raise UnknownRequestError("Options are post, patch, get, put, delete")

    return response
