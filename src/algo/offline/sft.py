from typing import Any

import torch

from algo.offline import OfflineTrainer


class SFTTrainer(OfflineTrainer):
    def loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return {"{split}/loss": outputs.loss}

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        metrics = self.loss(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.accelerator.backward(metrics["{split}/loss"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            return self.gather_metrics(metrics)

    def eval_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        metrics = self.loss(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return self.gather_metrics(metrics)
