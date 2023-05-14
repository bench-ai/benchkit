import os.path
from accelerate import Accelerator
from BenchKit.Miscellaneous.Settings import get_config


# def get_spt(x_axis_name: str,
#             y_axes_names: list[str],
#             version: int):
#
#     fl = get_config().get("code_versions")
#
#     if not fl:
#         raise ValueError("File path is missing. Consider migrating your code if it has not been done already")
#
#     code_dict = None
#     for i in fl:
#         if i.get("version") == version:
#             code_dict = i
#
#     if not code_dict:
#         raise ValueError(f"version {version} is not present")
#
#     file_name: str = os.path.split(code_dict["folder_location"])[-1]
#
#     file_name = file_name.split(".")[0]
#
#     return ScatterPlotTracker(x_axis_name, y_axes_names, file_name, version)


def save_checkpoint(acc: Accelerator):

    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")

    acc.save_state(output_dir="checkpoint")