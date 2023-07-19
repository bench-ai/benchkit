import copy
import json
from accelerate.tracking import GeneralTracker, on_main_process
import os
import pandas as pd
from BenchKit.Miscellaneous.User import update_server
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, ProjectConfiguration, MegatronLMPlugin, \
    GradientAccumulationPlugin, DeepSpeedPlugin


class AcceleratorInitializer:

    @staticmethod
    def accelerator(split_batches=False,
                    log_with=None,
                    even_batches: bool = True,
                    step_scheduler_with_optimizer: bool = True,
                    dynamo_backend: str = "no",
                    gradient_accumulation_steps: int = 1,
                    fsdp_plugin: FullyShardedDataParallelPlugin = None,
                    deepspeed_plugin: DeepSpeedPlugin = None,
                    megatron_lm_plugin: MegatronLMPlugin = None,
                    project_config: ProjectConfiguration | None = None,
                    gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
                    dispatch_batches: bool | None = None,
                    mixed_precision: str | None = None,
                    rng_types: list[str] | None = None,
                    kwargs_handlers: list | None = None) -> Accelerator:

        kwargs = {'split_batches': split_batches,
                  'log_with': log_with,
                  'even_batches': even_batches,
                  'step_scheduler_with_optimizer': step_scheduler_with_optimizer,
                  'dynamo_backend': dynamo_backend,
                  'gradient_accumulation_steps': gradient_accumulation_steps,
                  'mixed_precision': mixed_precision,
                  'rng_types': rng_types,
                  'kwargs_handlers': kwargs_handlers,
                  'fsdp_plugin': fsdp_plugin,
                  'deepspeed_plugin': deepspeed_plugin,
                  'megatron_lm_plugin': megatron_lm_plugin,
                  'project_config': project_config,
                  'gradient_accumulation_plugin': gradient_accumulation_plugin,
                  'dispatch_batches': dispatch_batches}

        for k, v in copy.deepcopy(kwargs).items():

            if not v:
                kwargs.pop(k)

        return Accelerator(**kwargs)


def get_tensorboard_tracker(config: dict[str: str],
                            *args,
                            **kwargs):

    kwargs["project_config"] = ProjectConfiguration(project_dir=".", logging_dir=os.getenv("LOG_DIR"))
    kwargs["log_with"] = "tensorboard"

    acc = AcceleratorInitializer.accelerator(**kwargs)
    acc.init_trackers(os.getenv("EXPERIMENT_NAME"), config=config)

    return acc, *acc.prepare(*args)


class BenchAccelerateTracker(GeneralTracker):
    name = "BaseTracker"
    requires_logging_directory = False

    @on_main_process
    def __init__(self,
                 run_name: str,
                 epochs):

        super().__init__()

        self._step = 0
        self._progress_bar = epochs
        self._instance = os.getenv("INSTANCE_ID")
        self.run_name = run_name

    @property
    def tracker(self):
        return self.run_name

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
            message: str | None = None,
            step: int | None = None):

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
