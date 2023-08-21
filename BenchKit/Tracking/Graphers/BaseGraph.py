class BenchGraph:

    def __init__(self,
                 graph_name: str,
                 config_id: dict):
        self.graph_name = graph_name.upper()
        self.config_id = config_id

    def init_graph(self):
        raise NotImplementedError("init_graph method must be added to current class")

    def log_value(self, *args, **kwargs):
        raise NotImplementedError("init_graph method must be added to current class")
