import requests


def upload_using_presigned_url(
    url: str, file_path: str, file_name: str, fields: dict
) -> None:
    """
    Uploads a file to the Bench Ai bucket, using a presigned post url

    :param url:
    :param file_path: The path where the file is saved
    :param file_name:
    :param fields: the fields returned with the presigned request
    """

    with open(file_path, "rb") as f:
        files = {"file": (file_name, f)}
        http_response = requests.post(url, data=fields, files=files)  # noqa S113

    if http_response.status_code != 204:
        raise RuntimeError(f"Failed to Upload {file_path}")
