import copy

from accelerate import Accelerator


# check out dispatch_batches

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
