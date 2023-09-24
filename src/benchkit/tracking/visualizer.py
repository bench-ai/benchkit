import concurrent.futures

from benchkit.misc.requests.graph import get_all_graphs
from benchkit.misc.requests.graph import get_time_series_points


class Plotter:
    def plot_time_series(self, ax, data_dict: dict, descriptor: dict):
        graph_name = descriptor["graph_name"]
        x_axis_name = descriptor["descriptors"]["x_name"]
        y_axis_name = descriptor["descriptors"]["y_name"]

        for i in data_dict:
            ln = i["line_name"]
            x_list = i["x_list"]
            y_list = i["y_list"]
            ax.plot(x_list, y_list, label=ln)

        ax.set_xlabel(x_axis_name)
        ax.set_ylabel(y_axis_name)
        ax.set_title(graph_name)
        ax.legend(loc="best")

    def __call__(self, graph_type, data_dict, descriptor, ax, **kwargs):
        match graph_type:
            case "TS":
                return self.plot_time_series(ax, data_dict, descriptor)
            case _:
                raise ValueError(f"Type: {graph_type}, is not supported")


class PointGetter:
    def __call__(self, tp: str, graph_instance: dict, *args, **kwargs):
        match tp:
            case "TS":
                return self.get_time_series_points(graph_instance["id"], graph_instance)
            case _:
                raise ValueError(f"Type: {tp}, is not supported")

    @staticmethod
    def get_time_series_points(graph_id: str, graph_parameters: dict):
        num_threads = 4
        ln = list(
            zip(
                graph_parameters["descriptors"]["line_names"],
                [graph_id] * len(graph_parameters["descriptors"]["line_names"]),
                strict=True,
            )
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_results = [executor.submit(get_time_series_points, *l) for l in ln]

            results = [future.result() for future in future_results]

        return results


def get_graph_n_points(config_id):
    p = PointGetter()

    graph_instance_list = get_all_graphs(config_id)

    type_list = [graph["graph_type"] for graph in graph_instance_list]

    desc_list = [
        {
            "graph_name": graph["name"],
            "graph_type": graph["graph_type"],
            "descriptors": graph["descriptors"],
        }
        for graph in graph_instance_list
    ]

    num_threads = 4

    zip_list = list(zip(type_list, graph_instance_list, strict=True))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_results = [executor.submit(p, *z) for z in zip_list]

        results = [
            (future.result(), desc)
            for future, desc in zip(future_results, desc_list, strict=True)
        ]

    return results


def get_display_matrix(plot_count: int) -> tuple[int, int]:
    x = 1
    y = 1
    for _ in range(plot_count):
        if (x * y) < plot_count:
            if y <= x:
                y += 1
            else:
                x += 1

    return x, y
