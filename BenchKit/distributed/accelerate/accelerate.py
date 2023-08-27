import os
from typing import Union
import torch
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, ProjectConfiguration, MegatronLMPlugin, \
    GradientAccumulationPlugin, DeepSpeedPlugin

from BenchKit.NeuralNetworks.evaluator import Evaluator


class BenchAccelerator(Accelerator):

    def __init__(self,
                 iteration: int = 0,
                 total_checkpoint_limit=3,
                 split_batches=False,
                 log_with=None,
                 even_batches: bool = True,
                 step_scheduler_with_optimizer: bool = True,
                 dynamo_backend: str = "no",
                 gradient_accumulation_steps: int = 1,
                 fsdp_plugin: FullyShardedDataParallelPlugin = None,
                 deepspeed_plugin: DeepSpeedPlugin = None,
                 megatron_lm_plugin: MegatronLMPlugin = None,
                 gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
                 dispatch_batches: bool | None = None,
                 mixed_precision: str | None = None,
                 rng_types: list[str] | None = None,
                 kwargs_handlers: list | None = None):

        project_config = ProjectConfiguration(iteration=iteration,
                                              project_dir=".",
                                              automatic_checkpoint_naming=True,
                                              total_limit=total_checkpoint_limit)

        super().__init__(project_config=project_config,
                         split_batches=split_batches,
                         log_with=log_with,
                         even_batches=even_batches,
                         step_scheduler_with_optimizer=step_scheduler_with_optimizer,
                         dynamo_backend=dynamo_backend,
                         gradient_accumulation_plugin=gradient_accumulation_plugin,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         fsdp_plugin=fsdp_plugin,
                         deepspeed_plugin=deepspeed_plugin,
                         megatron_lm_plugin=megatron_lm_plugin,
                         dispatch_batches=dispatch_batches,
                         mixed_precision=mixed_precision,
                         rng_types=rng_types,
                         kwargs_handlers=kwargs_handlers)

    def save_state(self,
                   pin_value: float = None,
                   evaluator: Evaluator = None,
                   **save_model_func_kwargs):

        if not pin_value:
            raise ValueError("pin_value must be provided")

        if not evaluator:
            raise ValueError("evaluator is required to save the state to the appropriate model")

        save_path = os.path.join(self.project_dir, "checkpoints", f"checkpoint_{self.save_iteration}")
        current_iteration = self.save_iteration

        super().save_state(**save_model_func_kwargs)

        # save_checkpoint(save_path, pin_value, self.save_iteration)

    def save_model(self,
                   model: torch.nn.Module,
                   save_directory: Union[str, os.PathLike],
                   max_shard_size: Union[int, str] = "10GB",
                   safe_serialization: bool = False):

        super().save_model(model,
                           save_directory,
                           max_shard_size=max_shard_size,
                           safe_serialization=safe_serialization)
