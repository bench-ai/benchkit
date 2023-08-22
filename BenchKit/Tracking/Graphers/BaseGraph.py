class BenchGraph:

    def __init__(self,
                 graph_name: str):
        self.graph_name = graph_name.upper()

    def init_graph(self, config_id):
        raise NotImplementedError("init_graph method must be added to current class")

    def log_value(self, *args, **kwargs):
        raise NotImplementedError("init_graph method must be added to current class")
