import abc
import time
from collections import defaultdict
from typing import Any

import torch
from accelerate.utils import AutocastKwargs
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from algo.base import BaseTrainer
from mixin import InstantiateTrainerDatasetMixin


class OfflineTrainer(BaseTrainer, InstantiateTrainerDatasetMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._instantiate_model(AutoModelForCausalLM)
        self._instantiate_tokenizer(relax_template=True)
        self._instantiate_dataloader()
        self._instantiate_optimizer()
        self._instantiate_scheduler()

        # accelerate
        self._initialize_acceleration()

        # resume training should be done after `accelerator.prepare()`
        if self.config.checkpoint:
            self.load_checkpoint(self.config.checkpoint)

    def _initialize_acceleration(self):
        # wrapping with accelerate
        self.logger.print(">>> Wrapping Engine <<<")
        wrappable_attributes = ("model",)
        if hasattr(self, "reference_model"):
            wrappable_attributes += ("reference_model",)
        wrappable_attributes += (
            "optimizer",
            "scheduler",
            "train_loader",
            "eval_loader",
        )

        self.model, *other_wrapped = self.accelerator.prepare(
            *map(lambda x: getattr(self, x), wrappable_attributes)
        )
        for attr, wrapped in zip(wrappable_attributes[1:], other_wrapped):
            setattr(self, attr, wrapped)

    @abc.abstractmethod
    def loss(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Compute loss for optimization."""

    @abc.abstractmethod
    def train_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Defines a single optimization step."""

    @abc.abstractmethod
    def eval_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Defines a single evaluation step."""

    def train(self):
        total_steps = self.config.epochs * len(self.train_loader)
        progress = tqdm(
            total=total_steps,
            position=0,
            leave=True,
            disable=not self.accelerator.is_main_process,
        )

        for epoch in range(self.config.epochs):
            self.set_mode(is_train=True)

            accumulated = defaultdict(list)
            for batch in self.train_loader:
                if self.batches_seen % self.config.eval_every == 0 and (
                    self.batches_seen > 0 or self.config.eval_first
                ):
                    eval_metrics = self.evaluate()
                    self.logger.log(eval_metrics, step=self.examples_seen)
                    if self.batches_seen > 0:
                        self.save_checkpoint(
                            f"step-{self.batches_seen}", eval_metrics
                        )

                start_time = time.time()

                self.set_mode(is_train=True)
                with self.accelerator.autocast(
                    AutocastKwargs(
                        self.model.training
                        and self.config.mixed_precision == "fp16"
                    )
                ):
                    with self.accelerator.accumulate(self.model):
                        # backward done in `train_step`
                        batch_metrics = self.train_step(batch)
                        for k, v in batch_metrics.items():
                            accumulated[k.format(split="train")].append(v)

                        # gather number of examples
                        num_examples = len(tuple(batch.values())[0])
                        num_examples = self.gather_metrics(
                            torch.tensor(
                                num_examples, device=self.accelerator.device
                            ),
                            reduce_op=torch.distributed.ReduceOp.SUM,
                        )

                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm,
                            )
                            mean_metrics = {
                                k: sum(v) / len(v) if len(v) > 0 else 0
                                for k, v in accumulated.items()
                            }
                            mean_loss = mean_metrics.get(
                                "train/loss", float("nan")
                            )

                            progress.set_description(
                                f"[Epoch {epoch + 1}/{self.config.epochs}] [Step {self.batches_seen}/{total_steps}] (loss {mean_loss:.4f})"
                            )
                            self.update_steps += 1

                            if self.batches_seen % self.config.log_every == 0:
                                mean_metrics.update(
                                    {
                                        "counter/lr": self.scheduler.get_last_lr()[
                                            0
                                        ],
                                        "counter/grad_norm": grad_norm,
                                        "counter/examples_per_sec": num_examples
                                        / (time.time() - start_time),
                                        "counter/update_steps": self.update_steps,
                                        "counter/batches_seen": self.batches_seen,
                                        "counter/examples_seen": self.examples_seen,
                                    }
                                )
                                self.logger.log(
                                    mean_metrics, step=self.examples_seen
                                )
                                accumulated.clear()

                        self.scheduler.step()
                        self.release_cuda_memory(*batch.values())

                progress.update(1)
                self.batches_seen += 1
                self.examples_seen += num_examples

            # save every epoch
            eval_metrics = self.evaluate()
            self.save_model(f"epoch-{epoch}", eval_metrics)

    def evaluate(self) -> dict[str, Any]:
        self.set_mode(is_train=False)
        accumulated = defaultdict(list)

        for _, batch in enumerate(
            tqdm(
                self.eval_loader,
                desc="Evaluating",
                disable=not self.accelerator.is_main_process,
            )
        ):
            with torch.inference_mode():
                batch_metrics = self.eval_step(batch)
                for k, v in batch_metrics.items():
                    accumulated[k.format(split="eval")].append(v)

        mean_metrics = {
            k: sum(v) / len(v) if len(v) > 0 else 0
            for k, v in accumulated.items()
        }
        return mean_metrics
