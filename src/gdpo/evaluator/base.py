import abc

from accelerate import Accelerator

from gdpo.config import EvaluateConfig


class BaseEvaluator(abc.ABC):
    config: EvaluateConfig

    def __init__(self, config: EvaluateConfig, accelerator: Accelerator) -> None:
        self.config = config
        self.accelerator = accelerator

    @abc.abstractmethod
    def evaluate(self) -> dict[str, any]:
        """Evaluates all samples with given criteria and returns summary statistics."""
        ...
