import os
from pathlib import Path

from BenchKit.Miscellaneous.requests.server import init_config
from typing import Callable

from BenchKit.tracking.save import upload_model_state, upload_model_save


class Config:
    """
    The configuration used for this model run
    """

    def __init__(self,
                 hyperparameters: dict,
                 evaluation_criteria: str,
                 mode: str):

        """
        :param hyperparameters: The values used to tweak this models performance
        :param evaluation_criteria: What criterion are you using to grade this model
        :param mode: options are `max` or `min` tells bench-kit if you want to optimize for the
        maximum or minimum of the evaluation_criteria
        """

        if len(hyperparameters) > 100:
            raise ValueError(f"hyperparameter length is {len(hyperparameters)}, the limit is 100")

        self.hyp_dict = {i.lower(): hyperparameters[i] for i in hyperparameters}
        self.evaluation_criteria = evaluation_criteria.lower()

        self.config_id = init_config(self.hyp_dict, self.evaluation_criteria)

        option_list = ["max", "min"]

        if mode.lower() not in option_list:
            raise ValueError(f"mode {mode} is not a valid option, available options are {option_list}")
        else:
            self.mode = mode

        self.best_evaluation_value = None
        self._last_iteration = -1

        root_folder_name = 'project_config_storage'
        config_folder_name = f"bench-{self.config_id}-config"

        self.config_save_path = Path(__file__).resolve().parent.parent / root_folder_name / config_folder_name

        os.makedirs(os.path.join(str(self.config_save_path), "checkpoints"),
                    exist_ok=True)

        os.makedirs(os.path.join(str(self.config_save_path), "save"),
                    exist_ok=True)

    @property
    def last_iteration(self):
        return self._last_iteration

    @last_iteration.setter
    def last_iteration(self, iteration: int):
        if self.last_iteration < iteration:
            raise ValueError(f"provided iteration: {iteration}, is less than last iteration {self.last_iteration}")
        else:
            self._last_iteration = iteration

    def should_save(self, eval_value: float) -> bool:
        return (self.mode == "max") == (eval_value > self.best_evaluation_value)

    def save_and_upload_state(self,
                              save_func: Callable[..., str],
                              iteration: int,
                              evaluation_value: float):

        state_dir_path: str = save_func()

        if not os.path.isdir(state_dir_path):
            raise RuntimeError(f"Function {save_func.__name__} did not return a path to a valid directory")

        upload_model_state(state_dir_path,
                           os.path.join(self.config_save_path, "checkpoints"),
                           iteration,
                           evaluation_value,
                           self)

    def save_and_upload_save(self,
                             save_func: Callable[..., str],
                             evaluation_value: float):

        save_dir_path: str = save_func()

        if not os.path.isdir(save_dir_path):
            raise RuntimeError(f"Function {save_func.__name__} did not return a path to a valid directory")

        upload_model_save(save_dir_path,
                          os.path.join(self.config_save_path, "save"),
                          evaluation_value,
                          self)

    def save_model_and_state(self,
                             save_state_func: Callable[..., str],
                             save_model_func: Callable[..., str],
                             iteration: int,
                             evaluation_value: float):

        self.last_iteration = iteration
        self.save_and_upload_state(save_state_func,
                                   iteration,
                                   evaluation_value)

        if self.should_save(evaluation_value):
            self.save_and_upload_save(save_model_func,
                                      evaluation_value)

