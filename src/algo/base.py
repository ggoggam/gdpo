import abc
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.utils.data.dataloader
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizer

from config import TrainConfig
from logger import MultipleLogger, RichLogger, WandbLogger
from mixin import ComputeMixin, GatherMetricsMixin, ReleaseMemoryMixin


class BaseTrainer(
    abc.ABC, ComputeMixin, GatherMetricsMixin, ReleaseMemoryMixin
):
    model: torch.nn.Module | PreTrainedModel
    tokenizer: PreTrainedTokenizer
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    train_loader: torch.utils.data.DataLoader
    eval_loader: torch.utils.data.DataLoader

    def __init__(
        self, config: TrainConfig, accelerator: Accelerator, *args, **kwargs
    ):
        self.config = config
        self.accelerator = accelerator

        loggers = [RichLogger()]
        if "wandb" in self.config.log_type and not self.config.debug:
            logger = WandbLogger(
                log_dir=self.config.log_dir,
                project_name=self.config.project_name,
                run_name=self.config.run_name,
            )
            loggers.append(logger)
        self.logger = MultipleLogger(loggers=loggers)

        self.logger.print(">>> Accelerator <<<")
        self.logger.print(f"{self.accelerator.state}")

        self.logger.log_config(dataclasses.asdict(self.config))

        self.update_steps = 0
        self.batches_seen = 0
        self.examples_seen = 0

    @property
    def total_steps(self) -> int:
        """Total number of steps."""
        return self.config.epochs * len(self.train_loader)

    @property
    def total_training_steps(self) -> int:
        """Total number of optimization steps."""
        update_steps_per_epoch = (
            len(self.train_loader) * self.config.gradient_accumulation_steps - 1
        ) // self.config.gradient_accumulation_steps
        return self.config.epochs * update_steps_per_epoch

    @property
    def warmup_steps(self) -> int:
        """ "Number of warmup steps."""
        if self.config.warmup_ratio:
            return int(self.config.warmup_ratio * self.total_training_steps)
        return self.config.warmup_steps

    @abc.abstractmethod
    def train(self) -> None: ...

    @abc.abstractmethod
    def evaluate(self) -> dict[str, Any]: ...

    def set_mode(self, is_train: bool = True):
        """Sets model to train or evaluation mode

        Args:
            train (bool, optional): Indicates training. Defaults to True.
        """
        if is_train:
            self.model.train()
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()

    def save_model(
        self, name: Optional[str] = None, metrics: dict[str, float] = {}
    ) -> None:
        """Saves model to the given path.

        Args:
            name (str, optional):
                Model name. Defaults to update steps if not given.
            metrics (dict[str, float], optional):
                Metrics to save with the model.
        """
        self.accelerator.wait_for_everyone()

        base_path = Path(self.config.log_dir) / self.config.run_name
        path = base_path / f"model-{self.update_steps}"
        if name is not None:
            path = base_path / f"model-{name}"

        self.logger.print(f">>> Saving model to {path}...")

        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod

        if isinstance(unwrapped, PreTrainedModel):
            unwrapped.save_pretrained(
                save_directory=path,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=self.accelerator.get_state_dict(unwrapped),
                safe_serialization=False,
            )
        else:
            self.accelerator.save(
                obj=unwrapped.state_dict(),
                f=path / "model.pt",
                safe_serialization=False,
            )

        if len(metrics) > 0 and self.accelerator.is_main_process:
            # dump metrics
            json.dump(metrics, open(path / "metrics.json", "w"), indent=2)

        # dump training config
        json.dump(
            dataclasses.asdict(self.config),
            open(path / "train_config.json", "w"),
            indent=2,
        )

    def save_checkpoint(
        self, name: Optional[str] = None, metrics: dict[str, float] = {}
    ) -> None:
        """Saves training checkpoint to the given path.
           When restoring, make sure to load in the same machine configuration.

        Args:
            name (str, optional):
                Checkpoint name. Defaults to update steps if not given.
            metrics (dict[str, float], optional):
                Metrics to save with the checkpoint.
        """
        self.accelerator.wait_for_everyone()

        base_path = Path(self.config.log_dir) / self.config.run_name
        path = base_path / f"checkpoint-{name}"
        if name is None:
            path = base_path / f"checkpoint-{self.update_steps}"

        self.logger.print(f">>> Saving checkpoint to {path}...")
        self.accelerator.save_state(path)

        if self.accelerator.is_main_process:
            state = {
                "update_steps": self.update_steps,
                "batches_seen": self.batches_seen,
                "examples_seen": self.examples_seen,
            }
            state |= dataclasses.asdict(self.config)
            json.dump(state, open(path / "trainer_state.json", "w"), indent=2)

            if len(metrics) > 0:
                json.dump(metrics, open(path / "metrics.json", "w"), indent=2)

    def load_checkpoint(self, path: Optional[os.PathLike] = None) -> None:
        """Loads training checkpoint from the given path.
           Must be loaded from checkpoint saved by `accelerator.save_checkpoint`.

        Args:
            path (os.PathLike, optional):
                Path to the checkpoint. Defaults to most recent checkpoint.
        """
        if path is None:
            base_path = Path(self.config.log_dir) / self.config.run_name
            # gets the latest checkpoint if no path is given
            checkpoints = list(base_path.glob("checkpoint-*"))
            if len(checkpoints) == 0:
                self.logger.print(">>> No checkpoint found... Skipping...")
                return
            path = max(checkpoints, key=os.path.getctime)

        self.logger.print(f">>> Loading checkpoint from {path}...")
        self.accelerator.load_state(path)

        if self.accelerator.is_main_process:
            with open(path / "trainer_state.json", "r") as f:
                state = json.load(f)
                self.update_steps = getattr(state, "update_states", 0)
                self.batches_seen = getattr(state, "batches_seen", 0)
                self.examples_seen = getattr(state, "examples_seen", 0)
