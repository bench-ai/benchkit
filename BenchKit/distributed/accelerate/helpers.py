from BenchKit.tracking.graphers.BaseGraph import BenchGraph
from .accelerate import BenchAccelerator
from .tracking.bench_tracker import BenchTracker
from datetime import timezone
from typing import Any
import datetime
import os


def get_bench_accelerator(*args,
                          **kwargs) -> tuple[BenchAccelerator, ...]:

    """
    Prepares all relevant objects and initializes a bench accelerator

    :param args: All the objects you wish to pass into Accelerator.prepare()
    :param kwargs: All the arguments you want to pass into the `BenchAccelerator`
    :return: The `BenchAccelerator` and all the prepared objects
    :rtype: BenchAccelerator, Any
    """

    acc = BenchAccelerator(**kwargs)

    return acc, *acc.prepare(*args)


def get_accelerator_with_bench_tracker(graph_list: list[BenchGraph],
                                       *args,
                                       **kwargs) -> tuple[BenchAccelerator, Any]:

    """
    Gets a BenchAccelerator initialized with the bench tracker

    :param graph_list: All the BenchGraphs you wish to log to
    :type graph_list: list[BenchGraph]
    :param args: All the objects you wish to pass into Accelerator.prepare()
    :param kwargs: All the arguments you want to pass into the `BenchAccelerator`
    :return: BenchAccelerator initialized with a BenchTracker
    :rtype: BenchAccelerator, Any
    """

    tracker = BenchTracker(*graph_list)
    kwargs["log_with"] = [tracker]

    result_tuple = get_bench_accelerator(*args, **kwargs)

    acc, prepared_objects = result_tuple[0], result_tuple[1:]

    project_name = os.getenv("EXPERIMENT_NAME").replace(" ",
                                                        "_") + str(datetime.datetime.now(timezone.utc)).replace(" ",
                                                                                                                "_")

    acc.init_trackers(project_name)
    return acc, *prepared_objects
