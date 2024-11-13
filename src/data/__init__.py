from abc import ABCMeta, abstractmethod
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        ...
