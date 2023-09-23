from benchkit.tracking.config import Config


class BenchGraph:
    def __init__(self, graph_name: str, config: Config):
        self.graph_name = graph_name.upper()
        self.config = config

    def init_graph(self):
        raise NotImplementedError("init_graph method must be added to current class")

    def log_value(self, *args, **kwargs):
        raise NotImplementedError("init_graph method must be added to current class")

    @property
    def config_id(self):
        return self.config.config_id
