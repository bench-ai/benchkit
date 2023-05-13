import json
from accelerate.tracking import GeneralTracker, on_main_process
import os
import pandas as pd


class BenchAccelerateTracker(GeneralTracker):
    name = "base"
    requires_logging_directory = False

    @on_main_process
    def __init__(self,
                 x_axis_name: str,
                 y_axes_name: list[str],
                 file_name: str,
                 increment_length: int,
                 version: int):

        super().__init__()

        self._x_axis_name = x_axis_name
        self._file_name = file_name
        self._version = version
        self._increment_length = increment_length

    @on_main_process
    def store_init_configuration(self, values: dict):
        if values.get("x_axis_name"):
            raise ValueError("Key x_axis_name")

        values["x_axis_name"] = self._x_axis_name

        with open(f"{self._file_name}.json", "w") as f:
            json.dump(values, f)

        # run started training signal
        # save run configuration to cloud
        # update progress bar

    @on_main_process
    def log(self, values: dict, step: int):

        ## fix this
        if not values["dataset_name"]:
            raise ValueError("dataset_name is required")

        if step < 0:
            raise ValueError("step cannot be negative")

        if not values:
            raise ValueError("the y value must be provided in values dictionary")

    @on_main_process
    def increment_progress(self, message: str):
        # send a signal to increment a progress bar on the home page
        pass

    @on_main_process
    def end_training(self):
        pass
        # save csv_results
        #
        # run successful kill signal


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
