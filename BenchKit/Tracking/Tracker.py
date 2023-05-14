import json
from accelerate.tracking import GeneralTracker, on_main_process
import os
import pandas as pd

from BenchKit.Miscellaneous.User import update_server


class BenchAccelerateTracker(GeneralTracker):
    name = "BaseTracker"
    requires_logging_directory = False

    @on_main_process
    def __init__(self,
                 epochs):

        super().__init__()

        self._step = 0
        self._progress_bar = epochs
        self._instance = os.getenv("INSTANCE_ID")

    @on_main_process
    def store_init_configuration(self,
                                 values: dict,
                                 message=None):

        if not message:
            message = json.dumps(values)[:248]

        update_server(self._instance,
                      progress=self._progress_bar,
                      last_message=message)

    @on_main_process
    def log(self,
            values: dict,
            message: str | None = None):

        if not message:
            message = json.dumps(values)[:248]

        self._step += 1

        update_server(self._instance,
                      current_step=self._step,
                      last_message=message)

    @on_main_process
    def end_training(self):
        update_server(self._instance,
                      last_message="TRAINING COMPLETED")


class BenchTracker:

    def __init__(self,
                 columns: list[str],
                 file_name: str,
                 hyperparameter_config: dict):

        if not os.path.isdir("log"):
            os.mkdir("log")

        self._save_path = os.path.join("log", file_name + ".csv")
        self._pd = pd.DataFrame(columns=columns, data={i: "0" for i in columns}, index=[0])
        self._first = True
        self._hp_config = hyperparameter_config

    @property
    def save_path(self):
        return self._save_path

    def write(self, *args, **kwargs):
        if self._first:
            self._pd.to_csv(path_or_buf=self._save_path,
                            index=False,
                            mode="w")
            self._first = False
        else:
            self._pd.to_csv(path_or_buf=self._save_path,
                            index=False,
                            mode="a+",
                            header=False)


class ScatterPlot(BenchTracker):

    def __init__(self,
                 x_axis: str,
                 y_axis_list: list[str],
                 file_name: str,
                 hyperparameter_config):
        columns = ["line_name", x_axis, *y_axis_list]
        self._axis_one = x_axis
        self._axis_list = y_axis_list

        super().__init__(columns,
                         file_name,
                         hyperparameter_config)

    def write(self,
              dataset_name: str,
              x: int,
              **kwargs):
        self._pd.iloc[0] = pd.Series(data={"line_name": dataset_name, self._axis_one: x, **kwargs})
        super().write()


if __name__ == '__main__':
    sc = ScatterPlot("my_x", ["loss", "correct_class"], "test", {"lr": 1})
    sc.write("train", 1, loss=10, correct_class_ratio=100)
    sc.write("train", 3, loss=None, correct_class_ratio=1)
