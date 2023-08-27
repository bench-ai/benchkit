from BenchKit.Miscellaneous.requests.server import init_config


class Config:

    def __init__(self,
                 hyperparameters: dict,
                 evaluation_criteria: str):

        if len(hyperparameters) > 100:
            raise ValueError(f"hyperparameter length is {len(hyperparameters)}, the limit is 100")

        self.hyp_dict = {i.lower(): hyperparameters[i] for i in hyperparameters}
        self.evaluation_criteria = evaluation_criteria.lower()

        self.config_id = init_config(self.hyp_dict, self.evaluation_criteria)