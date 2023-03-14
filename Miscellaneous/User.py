import json
import os
import pathlib
import requests
from pydantic.class_validators import wraps
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

            raise Credential("Unable to run method, credentials not verifiable")
        else:
            return response

    return wrapper


class Credential(Exception):
    pass


class UnknownRequest(Exception):
    pass


def get_current_user() -> dict:
    request_url = os.path.join(get_main_url(), "api", "auth", "user", "current")

    access, _ = AuthenticatedUser.read_credentials()
    response = request_executor("get",
                                url=request_url)

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
    else:
        raise UnknownRequest("Options are post, patch, get, put")

    return response


class AuthenticatedUser:

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
