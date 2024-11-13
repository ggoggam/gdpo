from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import optree
import torch
from accelerate import Accelerator
from torch import distributed as dist
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from common.config import TrainingConfig
from common.logger import Logger


class BaseTrainer(metaclass=ABCMeta):
    config: TrainingConfig
    accelerator: Accelerator
    logger: Logger

    model: Union[nn.Module, PreTrainedModel]
    tokenizer: PreTrainedTokenizer

    def __init__(self, config: TrainingConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.logger = Logger(
            log_type="none" if self.config.debug else self.config.log_type,
            log_dir=self.config.log_dir,
        )

    @abstractmethod
    def _init_model(self) -> None:
        ...

    @abstractmethod
    def _init_dataset(self) -> None:
        ...

    @abstractmethod
    def _init_engines(self) -> None:
        ...

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def evaluate(self):
        ...

    @staticmethod
    def compute_token_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        slide_mask: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor]:
        assert (
            logits.shape[:-1] == labels.shape
        ), "Logits and labels must have same shape in the first two dimensions"

        shifted_logps = (logits / temperature).log_softmax(dim=-1)[:, :-1, :]
        shifted_labels = labels[:, 1:].clone()
        mask = shifted_labels.not_equal(-100)
        shifted_labels[~mask] = 0

        logps = shifted_logps.gather(
            dim=-1, index=shifted_labels.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        if slide_mask:
            first_unmasked = torch.nonzero(mask.cumsum(dim=1) == 1)
            first_unmasked[:, -1] -= 1
            mask[first_unmasked[:, 0], first_unmasked[:, 1]] = True
        return logps * mask, mask

    @staticmethod
    def gather_metrics(
        metrics: dict[str, torch.Tensor], reduce_op: dist.ReduceOp = dist.ReduceOp.AVG
    ) -> dict[str, float]:
        """Gathers torch.Tensor across processes to main process

        Args:
            metrics (dict[str, Any]): metrics in tensor format
            reduce_op (dist.ReduceOp, optional): reduce operation. Defaults to dist.ReduceOp.AVG.

        Returns:
            dict[str, Any]: gathered metrics moved to cpu
        """
        if dist.is_initialized():
            for k in metrics:
                dist.all_reduce(metrics[k], reduce_op)
        metrics = optree.tree_map(lambda t: t.cpu().item(), metrics)
        return metrics

    def set_train_mode(self, mode: bool = True):
        if mode:
            self.model.train()
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()

    def set_eval_mode(self):
        self.set_train_mode(False)

    def save_model(self, name: Optional[str] = None) -> None:
        self.accelerator.wait_for_everyone()

        path = Path(self.config.log_dir) / f"model-{self.update_steps}"

    def save_checkpoint(self, name: Optional[str] = None) -> None:
        ...
