import abc
import multiprocessing

import torch
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer


class BaseUnpairedDataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    """Base dataset class that provides unpaired example."""

    dataset_name: str
    train_split: str
    eval_split: str

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 512,
        max_length: int = 1024,
    ):
        super().__init__()

        # num processes
        cpu_count = multiprocessing.cpu_count()
        num_proc = cpu_count // 2 if cpu_count > 1 else 1

        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_length = max_length

        self.dataset = dataset.map(self.preprocess_fn, num_proc=num_proc).map(
            self.apply_template, num_proc=num_proc
        )

    @abc.abstractmethod
    def preprocess_fn(self, example: dict[str, any]) -> dict[str, str]:
        """Preprocess a single example. Depends on the dataset. Applied before template.

        Args:
            example (dict[str, any]): a single example from the dataset.

        Returns:
            dict[str, str]: preprocessed example.
        """

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Returns the item at the given index."""
        return self.dataset[idx]

    def apply_template(self, example: dict[str, any]) -> dict[str, str]:
        """Apply chat template to the example.

        Args:
            example (dict[str, any]): a single example from the dataset.

        Returns:
            dict[str, str]: example with chat template applied.
        """
        prompt = self.tokenizer(
            self.tokenizer.apply_chat_template(
                conversation=example["chosen"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        chosen = self.tokenizer(
            self.tokenizer.apply_chat_template(
                conversation=[example["chosen"][-1]],
                tokenize=False,
                add_generation_prompt=False,
            ),
            add_special_tokens=False,
        )
        max_prompt_length = min(
            self.max_prompt_length, len(prompt["input_ids"])
        )
        max_response_length = self.max_length - max_prompt_length
        return {
            "history": example["chosen"][:-1],
            "prompt_input_ids": prompt["input_ids"][-max_prompt_length:],
            "prompt_attention_mask": prompt["attention_mask"][
                -max_prompt_length:
            ],
            "chosen_input_ids": chosen["input_ids"][:max_response_length],
            "chosen_attention_mask": chosen["attention_mask"][
                :max_response_length
            ],
        }

    def collate_fn(
        self, batch: list[dict[str, any]]
    ) -> dict[str, torch.Tensor]:
        """Collate function to provide to the dataloader.

        Args:
            batch (list[dict[str, any]]): a batch of examples.

        Returns:
            dict[str, torch.Tensor]: collated batch.
        """
        max_length = max(
            len(example["prompt_input_ids"] + example["chosen_input_ids"])
            for example in batch
        )
        # pad chosen
        self.tokenizer.padding_side = "right"
        chosen = self.tokenizer.pad(
            encoded_inputs={
                "input_ids": [
                    example["prompt_input_ids"] + example["chosen_input_ids"]
                    for example in batch
                ],
                "attention_mask": [
                    example["prompt_attention_mask"]
                    + example["chosen_attention_mask"]
                    for example in batch
                ],
            },
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        chosen_labels = pad_sequence(
            sequences=[
                torch.tensor(
                    len(example["prompt_attention_mask"]) * [-100]
                    + example["chosen_input_ids"],
                )
                for example in batch
            ],
            batch_first=True,
            padding_value=-100,
        )
        chosen_labels = torch.cat(
            (
                chosen_labels,
                chosen_labels.new_full(
                    (
                        chosen_labels.shape[0],
                        max_length - chosen_labels.shape[1],
                    ),
                    -100,
                ),
            ),
            dim=1,
        )
        return {
            "history": [example["history"] for example in batch],
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen_labels,
        }


class BasePromptOnlyDataset(BaseUnpairedDataset, metaclass=abc.ABCMeta):
    """Base dataset class used for generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_template(self, example: dict[str, any]) -> dict[str, str]:
        """Apply chat template to the example.

        Args:
            example (dict[str, any]): a single example from the dataset.

        Returns:
            dict[str, str]: example with chat template applied.
        """
        prompt = self.tokenizer(
            self.tokenizer.apply_chat_template(
                conversation=example["chosen"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        max_prompt_length = min(
            self.max_prompt_length, len(prompt["input_ids"])
        )
        return {
            "history": example["chosen"][:-1],
            "reference_response": example["chosen"][-1]["content"],
            "prompt_input_ids": prompt["input_ids"][-max_prompt_length:],
            "prompt_attention_mask": prompt["attention_mask"][
                -max_prompt_length:
            ],
        }

    def collate_fn(
        self, batch: list[dict[str, any]]
    ) -> dict[str, torch.Tensor]:
        """Collate function to provide to the dataloader.

        Args:
            batch (list[dict[str, any]]): a batch of examples.

        Returns:
            dict[str, torch.Tensor]: collated batch.
        """
        max_length = max(
            map(len, (example["prompt_input_ids"] for example in batch))
        )
        # pad chosen
        self.tokenizer.padding_side = "left"
        chosen = self.tokenizer.pad(
            encoded_inputs={
                "input_ids": [example["prompt_input_ids"] for example in batch],
                "attention_mask": [
                    example["prompt_attention_mask"] for example in batch
                ],
            },
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "history": [example["history"] for example in batch],
            "reference_response": [
                example["reference_response"] for example in batch
            ],
            "prompt_input_ids": chosen["input_ids"],
            "prompt_attention_mask": chosen["attention_mask"],
        }


class BasePairedDataset(BaseUnpairedDataset, metaclass=abc.ABCMeta):
    """Base dataset class that provides paired example."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_template(self, example):
        prompt = self.tokenizer(
            self.tokenizer.apply_chat_template(
                conversation=example["chosen"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        prompt_length = len(example["chosen"][:-1])
        # tokenize whole conversation for chosen and rejected
        chosen = self.tokenizer(
            self.tokenizer.apply_chat_template(
                conversation=[example["chosen"][-1]],
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
        max_prompt_length = min(
            self.max_prompt_length, len(prompt["input_ids"])
        )
        max_response_length = self.max_length - max_prompt_length
        return {
            "history": example["chosen"][:-1],
            "prompt_input_ids": prompt["input_ids"][-max_prompt_length:],
            "prompt_attention_mask": prompt["attention_mask"][
                -max_prompt_length:
            ],
            "chosen_input_ids": chosen["input_ids"][:max_response_length],
            "chosen_attention_mask": chosen["attention_mask"][
                :max_response_length
            ],
            "rejected_input_ids": rejected["input_ids"][:max_response_length],
            "rejected_attention_mask": rejected["attention_mask"][
                :max_response_length
            ],
        }

    def collate_fn(
        self, batch: list[dict[str, any]]
    ) -> dict[str, torch.Tensor]:
        # pad prompt from left
        max_length = max(
            max(
                len(example["prompt_input_ids"] + example["chosen_input_ids"])
                for example in batch
            ),
            max(
                len(example["prompt_input_ids"] + example["rejected_input_ids"])
                for example in batch
            ),
        )
        # pad chosen and rejected from right
        self.tokenizer.padding_side = "right"
        chosen = self.tokenizer.pad(
            encoded_inputs={
                "input_ids": [
                    example["prompt_input_ids"] + example["chosen_input_ids"]
                    for example in batch
                ],
                "attention_mask": [
                    example["prompt_attention_mask"]
                    + example["chosen_attention_mask"]
                    for example in batch
                ],
            },
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        chosen_labels = pad_sequence(
            sequences=[
                torch.tensor(
                    len(example["prompt_attention_mask"]) * [-100]
                    + example["chosen_input_ids"]
                )
                for example in batch
            ],
            batch_first=True,
            padding_value=-100,
        )
        chosen_labels = torch.cat(
            (
                chosen_labels,
                chosen_labels.new_full(
                    (
                        chosen_labels.shape[0],
                        max_length - chosen_labels.shape[1],
                    ),
                    fill_value=-100,
                ),
            ),
            dim=1,
        )
        rejected = self.tokenizer.pad(
            encoded_inputs={
                "input_ids": [
                    example["prompt_input_ids"] + example["rejected_input_ids"]
                    for example in batch
                ],
                "attention_mask": [
                    example["prompt_attention_mask"]
                    + example["rejected_attention_mask"]
                    for example in batch
                ],
            },
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_labels = pad_sequence(
            sequences=[
                torch.tensor(
                    len(example["prompt_attention_mask"]) * [-100]
                    + example["rejected_input_ids"]
                )
                for example in batch
            ],
            batch_first=True,
            padding_value=-100,
        )
        rejected_labels = torch.cat(
            (
                rejected_labels,
                rejected_labels.new_full(
                    (
                        rejected_labels.shape[0],
                        max_length - rejected_labels.shape[1],
                    ),
                    fill_value=-100,
                ),
            ),
            dim=1,
        )
        return {
            "history": [example["history"] for example in batch],
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected_labels,
        }
