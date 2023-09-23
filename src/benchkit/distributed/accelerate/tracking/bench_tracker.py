from accelerate.tracking import GeneralTracker
from accelerate.tracking import on_main_process

from benchkit.tracking.graphs.base_graph import BenchGraph


class BenchTracker(GeneralTracker):
    name = "BenchTracker"
    requires_logging_directory = False

    @on_main_process
    def __init__(self, *args: BenchGraph):
        super().__init__()
        self.tracker_list = {i.graph_name: i for i in args}

    @property
    def tracker(self):
        return self.tracker_list

    @on_main_process
    def store_init_configuration(self, values: dict):
        pass

    @on_main_process
    def log(self, values: dict, step: int | None = None):
        try:
            graph_name = values.pop("graph")
            graph_name = graph_name.upper()
        except KeyError as exc:
            raise KeyError(
                "There is no key provided in the values dict called graph_name"
            ) from exc

        self.tracker_list[graph_name].log_value(values, step=step)
