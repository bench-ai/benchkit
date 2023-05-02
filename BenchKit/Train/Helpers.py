import copy
import os.path
from pathlib import Path

from accelerate import Accelerator
def get_accelerator(split_batches=False,
                    even_batches: bool = True,
                    step_scheduler_with_optimizer: bool = True,
                    dynamo_backend: str = "no",
                    gradient_accumulation_steps: int = 1,
                    mixed_precision: str | None = None,
                    cpu: bool | None = None,
                    rng_types: list[str] | None = None,
                    kwargs_handlers: list | None = None) -> Accelerator:
    kwargs = {'split_batches': split_batches,
              'even_batches': even_batches,
              'step_scheduler_with_optimizer': step_scheduler_with_optimizer,
              'dynamo_backend': dynamo_backend,
              'gradient_accumulation_steps': gradient_accumulation_steps,
              'mixed_precision': mixed_precision,
              'cpu': cpu,
              'rng_types': rng_types,
              'kwargs_handlers': kwargs_handlers}

    for k, v in copy.deepcopy(kwargs).items():

        if not v:
            kwargs.pop(k)

    return Accelerator(**kwargs)


def wipe_temp(acc: Accelerator):
    from BenchKit.Data.Helpers import remove_all_temps
    acc.wait_for_everyone()
    remove_all_temps()
    acc.wait_for_everyone()


def write_script():
    template_path = Path(__file__).resolve().parent / "TrainScript.txt"

    if not os.path.isfile("TrainScript.py"):
        with open(template_path, "r") as read_file:

            with open("TrainScript.py", "w") as file:
                line = read_file.readline()
                while line:
                    file.write(line)
                    line = read_file.readline()

