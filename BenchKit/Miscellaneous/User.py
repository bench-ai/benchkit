import json
import os
import pathlib
from functools import wraps
import requests
from .Settings import get_main_url, get_credentials


def authorize_response(func):
    def run_request(*args, **kwargs) -> requests.Response:
        access, _ = AuthenticatedUser.read_credentials()
        header = {'Authorization': f"Bearer {access}"}
        kwargs.update({"headers": header})
        response: requests.Response = func(*args, **kwargs)
        return response

    @wraps(func)
    def wrapper(*args, **kwargs):

        response = run_request(*args, **kwargs)
        method = [AuthenticatedUser.refresh, AuthenticatedUser.login]

        if response.status_code != 200:

            for i in method:

                try:
                    i()
                except Credential:
                    pass

                response = run_request(*args, **kwargs)
                if response.status_code == 200:
                    return response

            raise Credential(f"Unable to run method, credentials not verifiable code:{response.status_code}")
        else:
            return response

    return wrapper


class Credential(Exception):
    pass


class UnknownRequest(Exception):
    pass


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
                                            "raw": True})

        if response.status_code == 200:
            return json.loads(response.content)["datasets"][0]
        else:
            raise RuntimeError("Unable to find Dataset")
    elif code == 200:
        return json.loads(response.content)
    else:
        raise RuntimeError("Unable to create Dataset")


def get_user_project(project_name: str) -> dict:
    request_url = os.path.join(get_main_url(), "api", "project", "user", "list")

    response = request_executor("get",
                                url=request_url,
                                params={"name": project_name,
                                        "page": 1})

    if response.status_code != 200:
        raise RuntimeError("Project does not exists please register it on bench")
    else:
        return json.loads(response.content)["projects"][0]


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

    @staticmethod
    def delete_credentials():
        current_credentials_path = pathlib.Path(__file__).resolve().parent / "Credentials.json"
        if os.path.exists(current_credentials_path):
            os.remove(current_credentials_path)

    @staticmethod
    def write_credentials(new_data: dict):
        current_credentials_path = pathlib.Path(__file__).resolve().parent / "Credentials.json"

        cred_dict = {}
        if os.path.isfile(current_credentials_path):
            access, refresh = AuthenticatedUser.read_credentials()
            cred_dict.update({"access_token": access,
                              "refresh_token": refresh})

        cred_dict.update(new_data)

        with open(current_credentials_path, "w") as file:
            json.dump(cred_dict, file)

    @staticmethod
    def read_credentials() -> tuple[str, str]:
        current_credentials_path = pathlib.Path(__file__).resolve().parent / "Credentials.json"

        curr_json = {}
        if not os.path.isfile(current_credentials_path):
            AuthenticatedUser.login()

        with open(current_credentials_path, "r") as file:
            curr_json.update(json.load(file))

        return curr_json["access_token"], curr_json["refresh_token"]

    @staticmethod
    def refresh():
        request_url = os.path.join(get_main_url(), "api", "auth", "token", "refresh")
        _, refresh = AuthenticatedUser.read_credentials()
        response = requests.get(url=request_url,
                                cookies={"refresh_token": refresh})

        if response.status_code != 200:
            raise Credential("Access Token could not be Refreshed try logging in again")
        else:
            access_token = json.loads(response.content)
            AuthenticatedUser.write_credentials(access_token)

    @staticmethod
    def login():
        request_url = os.path.join(get_main_url(), "api", "auth", "login")

        username, password = get_credentials()
        payload = {
            "username": username,
            "password": password
        }

        response = requests.post(request_url,
                                 json=payload)

        if response.status_code != 200:
            raise Credential("invalid username / password")
        else:
            content = json.loads(response.content)
            cred_dict = {"access_token": content["access_token"],
                         "refresh_token": response.cookies.get("refresh_token")}

            AuthenticatedUser.write_credentials(cred_dict)

    @staticmethod
    def logout():
        request_url = os.path.join(get_main_url(), "api", "auth", "logout")
        response = requests.post(request_url)

        if response.status_code != 200:
            raise Credential("Unable to logout")
        else:
            AuthenticatedUser.delete_credentials()
            print("...logged out")
