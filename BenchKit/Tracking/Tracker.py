import concurrent.futures
import copy
from accelerate.tracking import GeneralTracker, on_main_process
import os
from BenchKit.Miscellaneous.User import init_config, make_time_series_graph, plot_time_series_point
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, ProjectConfiguration, MegatronLMPlugin, \
    GradientAccumulationPlugin, DeepSpeedPlugin

from datetime import timezone
import datetime


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


def get_accelerator(config: dict[str: str],
                    *args,
                    **kwargs):
    config_id = init_config(config)
    acc = AcceleratorInitializer.accelerator(**kwargs)
    return config_id, acc, *acc.prepare(*args)


def get_time_series_tracker(config: dict[str: str],
                            graph_names: str | tuple[str, ...],
                            line_names: tuple[str, ...] | tuple[tuple[str, ...], ...],
                            x_axis_name: str | tuple[str, ...],
                            y_axis_name: str | tuple[str, ...],
                            *args,
                            **kwargs):

    config_id = init_config(config)
    ts = BenchAccelerateTimeSeriesTracker(graph_names,
                                          line_names,
                                          x_axis_name,
                                          y_axis_name,
                                          config_id)

    kwargs["log_with"] = [ts]

    acc = AcceleratorInitializer.accelerator(**kwargs)
    project_name = os.getenv("EXPERIMENT_NAME").replace(" ",
                                                        "_") + str(datetime.datetime.now(timezone.utc)).replace(" ",
                                                                                                                "_")

    acc.init_trackers(project_name, config=config)
    return config_id, acc, *acc.prepare(*args)


def get_tensorboard_tracker(config: dict[str: str],
                            *args,
                            **kwargs):
    config_id = init_config(config)

    kwargs["project_config"] = ProjectConfiguration(project_dir=".", logging_dir=os.getenv("LOG_DIR"))
    kwargs["log_with"] = "tensorboard"

    acc = AcceleratorInitializer.accelerator(**kwargs)
    project_name = os.getenv("EXPERIMENT_NAME").replace(" ",
                                                        "_") + str(datetime.datetime.now(timezone.utc)).replace(" ",
                                                                                                                "_")

    acc.init_trackers(project_name, config=config)
    return config_id, acc, *acc.prepare(*args)


def get_multi_tracker(config: dict[str: str],
                      tracker_list: list,
                      *args,
                      **kwargs):

    config_id = init_config(config)
    kwargs["log_with"] = tracker_list

    acc = AcceleratorInitializer.accelerator(**kwargs)
    project_name = os.getenv("EXPERIMENT_NAME").replace(" ",
                                                        "_") + str(datetime.datetime.now(timezone.utc)).replace(" ",
                                                                                                                "_")

    acc.init_trackers(project_name, config=config)
    return config_id, acc, *acc.prepare(*args)


class BenchAccelerateTimeSeriesTracker(GeneralTracker):
    name = "BenchTimeSeriesGraph"
    requires_logging_directory = False

    @on_main_process
    def __init__(self,
                 graph_names: str | tuple[str, ...],
                 line_names: tuple[str] | tuple[tuple[str, ...], ...],
                 x_axis_name: str | tuple[str, ...],
                 y_axis_name: str | tuple[str, ...],
                 config_id: str):

        super().__init__()
        self.graph_names = (graph_names,) if isinstance(graph_names, str) else graph_names
        line_names = (line_names,) if isinstance(line_names[0], str) else line_names
        self.config_id = config_id

        self.line_names = []
        for i in line_names:
            if len(i) > 50:
                raise ValueError("No more than 50 lines can be generated per graph")

            line_tup = []
            for j in range(len(i)):
                if not isinstance(i[j], str):
                    raise ValueError("line_names must be strings")

                line_tup.append(i[j].upper())

            self.line_names.append(line_tup)

        self.line_names = tuple(self.line_names)

        self.x_axis_names = (x_axis_name,) if isinstance(x_axis_name, str) else x_axis_name
        self.y_axis_names = (y_axis_name,) if isinstance(y_axis_name, str) else y_axis_name
        self.graph_id_dict = {}

    @property
    def tracker(self):
        return self.graph_id_dict

    @on_main_process
    def store_init_configuration(self,
                                 values: dict):

        config_id_list = [self.config_id] * len(self.graph_names)

        zips = zip(config_id_list,
                   self.graph_names,
                   self.line_names,
                   self.x_axis_names,
                   self.y_axis_names)

        num_threads = 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_results = [executor.submit(make_time_series_graph, *z) for z in zips]

            results = [future.result() for future in future_results]

        for r, g in zip(results, self.graph_names):
            self.graph_id_dict[g.upper()] = r

    @on_main_process
    def log(self,
            values: dict,
            step: int | None = None,
            graph_name: str | None = None):

        graph_name = graph_name.upper()

        try:
            graph_id = self.graph_id_dict[graph_name]

        except KeyError:
            raise KeyError(f"There is no graph named {graph_name}")

        num_threads = 4

        graph_id_list = [graph_id] * len(values)
        line_name_list = list(values.keys())
        x_value_list = [step] * len(values)
        y_value_list = list(values.values())

        zips = zip(graph_id_list, line_name_list, x_value_list, y_value_list)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_results = [executor.submit(plot_time_series_point, *z) for z in zips]

            for future in future_results:
                future.result()
