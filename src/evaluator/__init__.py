from abc import ABCMeta, abstractmethod
from typing import Any, Union

from accelerate import Accelerator
from datasets import Dataset as HFDataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel

from config import EvaluateConfig


class Evaluator(metaclass=ABCMeta):
    config: EvaluateConfig
    accelerator: Accelerator
    model: Union[nn.Module, PreTrainedModel]
    loaders: dict[str, Union[HFDataset, Dataset, DataLoader]]

    def __init__(self, config: EvaluateConfig, accelerator: Accelerator) -> None:
        self.config = config
        self.accelerator = accelerator

    @abstractmethod
    def _init_model(self):
        """Initializes models"""
        ...

    @abstractmethod
    def _init_loaders(self):
        """Initializes data loaders"""
        ...

    @abstractmethod
    def _init_engine(self):
        """Wraps models and data loaders with hardware acceleration"""
        ...

    @abstractmethod
    def evaluate_sample(self, *args, **kwargs) -> dict[str, Any]:
        """Evaluates a single sample with specified criteria."""
        ...

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Evaluates all samples with given criteria and returns summary statistics."""
        ...
