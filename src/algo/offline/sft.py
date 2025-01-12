import torch

from algo.offline.base import OfflineTrainer
from mixin.instantiate import InstantiateModelMixin


class SFTTrainer(OfflineTrainer, InstantiateModelMixin):
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

    def train_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        metrics = self.loss(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            labels=batch["chosen_labels"],
        )
        self.accelerator.backward(metrics["{split}/loss"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            return self.gather_metrics(metrics)

    def eval_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        metrics = self.loss(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            labels=batch["chosen_labels"],
        )
        return self.gather_metrics(metrics)
