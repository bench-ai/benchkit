import json
from accelerate.tracking import GeneralTracker, on_main_process
import os
import pandas as pd


class ScatterPlotTracker(GeneralTracker):
    name = "ScatterPlot"
    requires_logging_directory = False

    @on_main_process
    def __init__(self,
                 x_axis_name: str,
                 y_axes_name: list[str],
                 file_name: str,
                 version: int):

        super().__init__()
        self._scatter_plot = ScatterPlot(x_axis_name,
                                         y_axes_name,
                                         file_name)

        self._x_axis_name = x_axis_name
        self._file_name = file_name
        self._version = version

    @property
    def tracker(self):
        return self._scatter_plot

    @on_main_process
    def store_init_configuration(self, values: dict):
        if values.get("x_axis_name"):
            raise ValueError("Key x_axis_name")

        values["x_axis_name"] = self._x_axis_name

        with open(f"{self._file_name}.json", "w") as f:
            json.dump(values, f)

    @on_main_process
    def log(self, values: dict, step: int):
        if not values["dataset_name"]:
            raise ValueError("dataset_name is required")

        if step < 0:
            raise ValueError("step cannot be negative")

        if not values:
            raise ValueError("the y value must be provided in values dictionary")

        self._scatter_plot.write(values["dataset_name"],
                                 step,
                                 **values)

    @on_main_process
    def end_training(self):
        pass


class BenchFrame:

    def __init__(self,
                 columns: list[str],
                 file_name: str):

        if not os.path.isdir("log"):
            os.mkdir("log")

        self._save_path = os.path.join("log", file_name + ".csv")

        self._pd = pd.DataFrame(columns=columns, data={i: "0" for i in columns}, index=[0])
        self._first = True

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


class ScatterPlot(BenchFrame):

    def __init__(self,
                 x_axis: str,
                 y_axis_list: list[str],
                 file_name: str):
        columns = ["line_name", x_axis, *y_axis_list]
        self._axis_one = x_axis
        self._axis_list = y_axis_list

        super().__init__(columns, file_name)

    def write(self,
              dataset_name: str,
              x: int,
              **kwargs):
        self._pd.iloc[0] = pd.Series(data={"line_name": dataset_name, self._axis_one: x, **kwargs})
        super().write()


if __name__ == '__main__':
    sc = ScatterPlot("my_x", ["loss", "correct_class"], "test")
    sc.write("train", 1, loss=10, correct_class_ratio=100)
    sc.write("train", 3, loss=None, correct_class_ratio=1)
