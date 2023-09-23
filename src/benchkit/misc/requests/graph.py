import json
import os

from .user import request_executor
from benchkit.misc.settings import get_main_url


def plot_time_series_point(graph_id: str, line_name: str, x_value: int, y_value: float):
    request_url = os.path.join(get_main_url(), "api", "tracking", "time-series", "plot")

    response = request_executor(
        "post",
        url=request_url,
        json={
            "graph_id": graph_id,
            "data_point": {
                "line_name": line_name,
                "x_value": x_value,
                "y_value": y_value,
            },
        },
    )
    response_dict = json.loads(response.content)

    return response_dict


def make_time_series_graph(
    config_id: str, name: str, line_names: list[str], x_name: str, y_name: str
) -> str:
    request_url = os.path.join(get_main_url(), "api", "tracking", "time-series")

    response = request_executor(
        "post",
        url=request_url,
        json={
            "config_id": config_id,
            "name": name,
            "descriptors": {
                "line_names": line_names,
                "x_name": x_name,
                "y_name": y_name,
            },
        },
    )

    response_dict = json.loads(response.content)

    return response_dict["id"]


def get_all_graphs(config_id: str):
    next_page = 1
    graph_list = []

    request_url = os.path.join(get_main_url(), "api", "tracking", "graphs", "b-k")

    while next_page:
        response = request_executor(
            "get", url=request_url, params={"page": next_page, "config_id": config_id}
        )

        response_list = json.loads(response.content)

        next_page = response_list["next_page"]

        graph_list += response_list["graph_list"]

    return graph_list


def get_time_series_points(line_name: str, graph_id):
    point_count = 100

    request_url = os.path.join(
        get_main_url(), "api", "tracking", "time-series", "plot", "segmented", "b-k"
    )

    response = request_executor(
        "get",
        url=request_url,
        params={
            "point_count": point_count,
            "line_name": line_name,
            "graph_id": graph_id,
        },
    )

    response_content = json.loads(response.content)

    response_dict = {"line_name": line_name, "x_list": [], "y_list": []}

    for i in response_content["line_list"]:
        response_dict["x_list"].append(i["data"]["x_value"])
        response_dict["y_list"].append(i["data"]["y_value"])

    return response_dict
