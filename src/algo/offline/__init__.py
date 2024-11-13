import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import asdict
from typing import Any

import torch
from accelerate import Accelerator
from tqdm import tqdm

from algo import BaseTrainer
from common.config import TrainingConfig


class OfflineTrainer(BaseTrainer):
    def __init__(self, config: TrainingConfig, accelerator: Accelerator):
        super().__init__(config, accelerator)

        self.logger.print(">>> Accelerator")
        self.logger.print(f"{self.accelerator.state}")

        self.logger.print(">>> Configurations")
        self.logger.print(asdict(config))

        self.logger.print(">>> Model Initialization")
        self._init_model()
        self.logger.print(">>> Dataset Initialization")
        self._init_dataset()
        self.logger.print(">>> Wrapping Engine")
        self._init_engines()

        self.update_steps = 0
        self.batches_seen = 0
        self.examples_seen = 0

    @abstractmethod
    def loss(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        ...

    @abstractmethod
    def eval_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        ...

    def train(self):
        progress = tqdm(
            total=self.config.epochs * len(self.train_loader),
            position=0,
            leave=True,
            disable=not self.accelerator.is_main_process,
        )

        for epoch in range(self.config.epochs):
            accumulated = defaultdict(list)

            for batch in self.train_loader:

                if self.batches_seen % self.config.eval_every == 0 and (
                    self.batches_seen > 0 or self.config.eval_first
                ):
                    eval_metrics = self.eval()

                    self.logger.log(eval_metrics, step=self.examples_seen)

                    if self.batches_seen > 0:
                        self.save_model(f"step-{self.batches_seen}")

                start_time = time.time()
