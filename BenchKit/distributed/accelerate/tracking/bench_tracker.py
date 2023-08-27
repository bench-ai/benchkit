from accelerate.tracking import GeneralTracker, on_main_process
from BenchKit.tracking.graphs.base_graph import BenchGraph
import concurrent.futures


class BenchTracker(GeneralTracker):
    name = "BenchTracker"
    requires_logging_directory = False

    @on_main_process
    def __init__(self,
                 *args: BenchGraph):

        super().__init__()
        self.tracker_list = {i.graph_name: i for i in args}

    @property
    def tracker(self):
        return self.tracker_list

    @on_main_process
    def store_init_configuration(self,
                                 values: dict):

        num_threads = 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_results = [executor.submit(graph.init_graph) for graph in self.tracker_list.values()]

            [future.result() for future in future_results]

    @on_main_process
    def log(self,
            values: dict,
            step: int | None = None):

        try:
            graph_name = values.pop("graph")
            graph_name = graph_name.upper()
        except KeyError:
            raise KeyError(f"There is no key provided in the values dict called graph_name")

        self.tracker_list[graph_name].log_value(values, step=step)
