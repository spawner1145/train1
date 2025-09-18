# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from lightning_utilities.core.imports import module_available
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.strategy import _BackwardSyncControl
from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
from fairscale.optim import OSS


class DDPShardedStrategy(DDPStrategy):
    """Optimizer and gradient sharded training provided by FairScale."""

    _REDUCE_BUFFER_SIZE_DEFAULT: int = 2**23  # 8M

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
            process_group_backend=process_group_backend,
            timeout=timeout,
            **kwargs,
        )
        self._backward_sync_control = _FairscaleBackwardSyncControl()
        if "reduce_buffer_size" not in self._ddp_kwargs:
            # For multi-node training, enabling bucketing will improve performance.
            self._ddp_kwargs["reduce_buffer_size"] = self._REDUCE_BUFFER_SIZE_DEFAULT if self.num_nodes > 1 else 0

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model and optimizers with fairscale components.

        Return:
            The model wrapped into a :class:`~fairscale.nn.data_parallel.ShardedDataParallel` module
            and a list of optimizer wrapped in :class:~`fairscale.optim.OSS`.
        """
        optimizers = _reinit_optimizers_with_oss(optimizers, self.precision, self.num_nodes)
        for optimizer in optimizers:
            # This forces buckets to be rebuilt on the first forward pass
            # We are not sure why this is needed, but it prevents an error resulting from buckets having a different
            # device than the params
            optimizer._clear_cache()
        model = ShardedDataParallel(module, sharded_optimizer=optimizers, **self._ddp_kwargs)
        return model, optimizers

    def setup_module(self, module: Module) -> DistributedDataParallel:
        """Setting up the module without optimizers in this strategy is not supported.

        Please use :meth:`setup_module_and_optimizers` instead.
        """
        raise NotImplementedError(self._err_msg_joint_setup_required())

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Optimizers can only be set up jointly with the model in this strategy.

        Please use :meth:`setup_module_and_optimizers` to set up both module and optimizer(s) together.
        """
        raise NotImplementedError(self._err_msg_joint_setup_required())

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_sharded",
            cls,
            description=cls.__class__.__name__,
        )
        strategy_registry.register("ddp_sharded_spawn", cls, description=cls.__class__.__name__, start_method="spawn")


def _reinit_optimizers_with_oss(optimizers: List[Optimizer], precision: Precision, num_nodes: int) -> List["OSS"]:
    for x, optimizer in enumerate(optimizers):
        if not isinstance(optimizer, OSS):
            optim_class = type(optimizer)
            zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
            is_fp16 = precision.precision == "16" or precision.precision == "16-mixed"
            # For multi-node training, compressing the model shards in fp16 before broadcasting
            # improves performance. When using PyTorch AMP, it will not degrade
            # the model performance.
            zero_optimizer.broadcast_fp16 = is_fp16 and num_nodes > 1
            optimizers[x] = zero_optimizer
            del optimizer
    return optimizers


class _FairscaleBackwardSyncControl(_BackwardSyncControl):
    @contextmanager
    def no_backward_sync(self, module: Module) -> Generator:
        """Blocks gradient synchronization inside the :class:`~fairscale.nn.data_parallel.ShardedDataParallel`
        wrapper."""
        if not isinstance(module, ShardedDataParallel):
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{self.__class__.__name__}.no_backward_sync` is wrapped in `ShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        with module.no_sync():
            yield None


def _optimizer_has_flat_params(optimizer: Optimizer) -> bool:
    from fairscale.nn.misc.flatten_params_wrapper import FlatParameter

    return any(isinstance(param, FlatParameter) for param in optimizer.param_groups[0]["params"])