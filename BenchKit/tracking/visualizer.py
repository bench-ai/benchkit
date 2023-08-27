from BenchKit.Miscellaneous.requests.server import get_all_configs
from BenchKit.Miscellaneous.requests.graph import get_all_graphs, get_time_series_points
from tabulate import tabulate
import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt


class Plotter:

    def plot_time_series(self,
                         ax,
                         data_dict: dict,
                         descriptor: dict):

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
        ax.legend(loc='best')

    def __call__(self,
                 graph_type,
                 data_dict,
                 descriptor,
                 ax,
                 **kwargs):

        match graph_type:
            case "TS":
                return self.plot_time_series(ax, data_dict, descriptor)
            case _:
                raise ValueError(f"Type: {graph_type}, is not supported")


class PointGetter:

    def __call__(self,
                 tp: str,
                 graph_instance: dict,
                 *args,
                 **kwargs):

        match tp:
            case "TS":
                return self.get_time_series_points(graph_instance["id"],
                                                   graph_instance)
            case _:
                raise ValueError(f"Type: {tp}, is not supported")

    def get_time_series_points(self,
                               graph_id: str,
                               graph_parameters: dict):

        num_threads = 4
        ln = list(zip(graph_parameters["descriptors"]["line_names"],
                      [graph_id] * len(graph_parameters["descriptors"]["line_names"])))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_results = [executor.submit(get_time_series_points, *l) for l in ln]

            results = [future.result() for future in future_results]

        return results


def get_graph_n_points(config_id):
    p = PointGetter()

    graph_instance_list = get_all_graphs(config_id)

    type_list = [graph["graph_type"] for graph in graph_instance_list]

    desc_list = [
        {"graph_name": graph["name"],
         "graph_type": graph["graph_type"],
         "descriptors": graph["descriptors"]} for graph in graph_instance_list
    ]

    num_threads = 4

    zip_list = list(zip(type_list, graph_instance_list))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_results = [executor.submit(p, *z) for z in zip_list]

        results = [(future.result(), desc) for future, desc in zip(future_results, desc_list)]

    return results


def get_display_matrix(plot_count: int) -> tuple[int, int]:

    x = 1
    y = 1
    for i in range(plot_count):
        if (x * y) < plot_count:
            if y <= x:
                y += 1
            else:
                x += 1

    return x, y


def display_all_configs(instance_id):
    p = Plotter()
    config_list, id_list = get_all_configs(instance_id)

    config_df = pd.DataFrame(data=config_list)

    print(tabulate(config_df, headers='keys', tablefmt='psql', showindex=True))

    loop = True

    while loop:
        inp = input("Enter the number of the model, whose graphs you wish to see. Type any other key to exit: ")

        try:
            inp = int(inp)
            c_id = id_list[inp]

            graph_list = get_graph_n_points(c_id)

            rows, columns = get_display_matrix(len(graph_list))

            fig, axs = plt.subplots(rows, columns, squeeze=False)

            mx = max(rows, columns)

            for idx, (data_dict, graph_description) in enumerate(graph_list):

                x = idx // mx
                y = idx % mx

                p(graph_description["graph_type"],
                  data_dict,
                  graph_description,
                  axs[x, y])

            plt.show()

        except ValueError:
            loop = False


def display_time_series():
    pass
