import concurrent.futures
import copy
from accelerate.tracking import GeneralTracker, on_main_process
import os
from BenchKit.Miscellaneous.User import init_config
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, ProjectConfiguration, MegatronLMPlugin, \
    GradientAccumulationPlugin, DeepSpeedPlugin

from datetime import timezone
import datetime

from .Graphers.TimeSeries import BenchGraph


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


def get_bench_tracker(config: dict[str: str],
                      graph_list: list[BenchGraph],
                      *args,
                      **kwargs):

    config_id = init_config(config)

    tracker = BenchTracker(config_id, *graph_list)
    kwargs["log_with"] = [tracker]

    acc = AcceleratorInitializer.accelerator(**kwargs)
    project_name = os.getenv("EXPERIMENT_NAME").replace(" ",
                                                        "_") + str(datetime.datetime.now(timezone.utc)).replace(" ",
                                                                                                                "_")

    acc.init_trackers(project_name, config=config)
    return config_id, acc, *acc.prepare(*args)


class BenchTracker(GeneralTracker):
    name = "BenchTracker"
    requires_logging_directory = False

    @on_main_process
    def __init__(self,
                 config_id: str,
                 *args: BenchGraph):

        super().__init__()
        self.tracker_list = {i.graph_name: i for i in args}
        self.config_id = config_id

    @property
    def tracker(self):
        return self.tracker_list

    @on_main_process
    def store_init_configuration(self,
                                 values: dict):

        num_threads = 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_results = [executor.submit(graph.init_graph, self.config_id) for graph in self.tracker_list.values()]

            [future.result() for future in future_results]

    @on_main_process
    def log(self,
            values: dict,
            step: int | None = None):

        try:
            graph_name = values.pop("graph")
            graph_name = graph_name.upper()
        except KeyError:
            raise KeyError(f"There is no key provided in the values dict called graph_name")

        self.tracker_list[graph_name].log_value(values, step=step)
