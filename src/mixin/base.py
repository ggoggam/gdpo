from typing import Protocol, Union

import torch
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizer

from config import CommonConfig


class TrainerMixinProtocol(Protocol):
    """Protocol for better typing support in mixin classes."""

    @property
    def config(self) -> CommonConfig: ...

    @property
    def accelerator(self) -> Accelerator: ...

    @property
    def model(self) -> Union[torch.nn.Module, PreTrainedModel]: ...

    @property
    def tokenizer(self) -> PreTrainedTokenizer: ...

    @property
    def optimizer(self) -> torch.optim.Optimizer: ...

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler: ...


class TrainerWithReferenceModelMixinProtocol(TrainerMixinProtocol):
    """Protocol for better typing support in mixin classes."""

    @property
    def reference_model(self) -> Union[torch.nn.Module, PreTrainedModel]: ...
