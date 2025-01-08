import abc
import dataclasses

import torch
from torch.utils.data import IterableDataset


@dataclasses.dataclass
class Example:
    # conversation history without the last response
    history: list[dict[str, str]]

    def __init__(self):
        raise NotImplementedError("Cannot instantiate abstract base class `Example`")


@dataclasses.dataclass
class UnpairedExample(Example):
    """Unpaired example for training. Used for supervised fine-tuning.'

    Args:
        response (str): response to the conversation history
        input_ids (torch.LongTensor): input ids for the response
        attention_mask (torch.LongTensor): attention mask for the response
        labels (torch.LongTensor): labels for the response
    """

    response: str
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor


@dataclasses.dataclass
class PairedExample(Example):
    """Paired example for training. Used for preference optimization.

    Args:
        chosen (UnpairedExample): chosen example
        rejected (UnpairedExample): rejected example
    """

    chosen: UnpairedExample
    rejected: UnpairedExample


class BaseIterableDataset(IterableDataset, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    @property
    def name(self) -> str:
        """Name of the dataset."""
        ...

    @abc.abstractmethod
    def _apply_template(self, example: dict[str, str]) -> Example:
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...


class UnpairedDataset(BaseIterableDataset):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "unpaired"

    def _apply_template(self, example: dict[str, str]) -> UnpairedExample:
        prompt = self.tokenizer(
            self.tokenizer.apply_chat_template(
                conversation=example["chosen"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        prompt_length = len(example["chosen"][:-1])
        chosen = self.tokenizer(
            self.tokenizer.apply_chat_template(
                conversation=example["chosen"][-1],
                tokenize=False,
                add_generation_prompt=False,
            ),
            add_special_tokens=False,
        )
        rejected = self.tokenizer(
            self.tokenizer.apply_chat_template(
                conversation=example["rejected"][prompt_length:],
                tokenize=False,
                add_generation_prompt=False,
            ),
            add_special_tokens=False,
        )
        return UnpairedExample(
            history=example["chosen"][:-1],
            response=example["chosen"][-1],
            input_ids=chosen["input_ids"],
            attention_mask=chosen["attention_mask"],
            labels=chosen["input_ids"],
        )

    def __iter__(self):
        return self.iterable
