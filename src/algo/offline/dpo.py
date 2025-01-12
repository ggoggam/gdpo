import math
from typing import Literal, Union

import torch
from torch.nn import functional as F
from transformers import PreTrainedModel

from algo.offline.base import OfflineTrainer
from mixin.instantiate import InstantiateModelMixinWithReferenceModel


class DPOTrainer(OfflineTrainer, InstantiateModelMixinWithReferenceModel):
    reference_model: Union[torch.nn.Module, PreTrainedModel]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = self.config.algorithm.beta
        self.label_smoothing = self.config.algorithm.label_smoothing
        self.loss_type = self.config.algorithm.loss_type

    def _concat_forward(
        self,
        model: PreTrainedModel,
        chosen_input_ids: torch.LongTensor,
        chosen_attention_mask: torch.LongTensor,
        chosen_labels: torch.LongTensor,
        rejected_input_ids: torch.LongTensor,
        rejected_attention_mask: torch.LongTensor,
        rejected_labels: torch.LongTensor,
        reduce: Literal["none", "mean", "sum"] = "sum",
    ) -> tuple[torch.FloatTensor]:
        """Concatenates chosen and rejected response in batch dimension for faster training.

        Args:
            model (PreTrainedModel): model to forward
            chosen_input_ids (torch.LongTensor): chosen input ids
            chosen_attention_mask (torch.LongTensor): chosen attention mask
            chosen_labels (torch.LongTensor): chosen labels (offline trajectory)
            rejected_input_ids (torch.LongTensor): rejected input ids
            rejected_attention_mask (torch.LongTensor): rejected attention mask
            rejected_labels (torch.LongTensor): rejected labels (offline trajectory)
            reduce (Literal["none", "mean", "sum"], optional): reduction in vocab dimension. Defaults to "sum".

        Returns:
            tuple[torch.FloatTensor]: tuple of (chosen_logits, rejected_logits, chosen_logps, rejected_logps)
        """
        # concatenate in batch dim
        concat_input_ids = torch.cat(
            (chosen_input_ids, rejected_input_ids), dim=0
        )
        concat_attention_mask = torch.cat(
            (chosen_attention_mask, rejected_attention_mask), dim=0
        )
        concat_labels = torch.cat((chosen_labels, rejected_labels), dim=0)
        # concatenated forward
        logits: torch.FloatTensor = model(
            input_ids=concat_input_ids, attention_mask=concat_attention_mask
        )["logits"]
        logps, mask = self.compute_token_logps(
            logits=logits, labels=concat_labels
        )
        # reduce logps
        if reduce == "mean":
            logps = logps.sum(dim=-1) / mask.sum(dim=-1)
        elif reduce == "sum":
            logps = logps.sum(dim=-1)
        # split
        chosen_logits, rejected_logits = logits.chunk(chunks=2, dim=0)
        chosen_logps, rejected_logps = logps.chunk(chunks=2, dim=0)
        return (
            chosen_logits,
            rejected_logits,
            chosen_logps,
            rejected_logps,
        )

    def loss(
        self,
        chosen_input_ids: torch.LongTensor,
        chosen_attention_mask: torch.LongTensor,
        chosen_labels: torch.LongTensor,
        rejected_input_ids: torch.LongTensor,
        rejected_attention_mask: torch.LongTensor,
        rejected_labels: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        (
            *_,
            policy_chosen_logps,
            policy_rejected_logps,
        ) = self._concat_forward(
            model=self.model,
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=rejected_attention_mask,
            rejected_labels=rejected_labels,
            reduce="mean" if self.loss_type == "ipo" else "sum",
        )
        with torch.no_grad():
            (
                *_,
                ref_chosen_logps,
                ref_rejected_logps,
            ) = self._concat_forward(
                model=self.reference_model,
                chosen_input_ids=chosen_input_ids,
                chosen_attention_mask=chosen_attention_mask,
                chosen_labels=chosen_labels,
                rejected_input_ids=rejected_input_ids,
                rejected_attention_mask=rejected_attention_mask,
                rejected_labels=rejected_labels,
                reduce="mean" if self.loss_type == "ipo" else "sum",
            )

        logits = (policy_chosen_logps - policy_rejected_logps) - (
            ref_chosen_logps - ref_rejected_logps
        )
        if self.loss_type == "ipo":
            loss = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "sigmoid":
            loss = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "normcdf":
            loss = -torch.log(
                0.5 * (1 + torch.erf(self.beta * logits / math.sqrt(2)))
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.sum(dim=-1) - ref_chosen_logps.sum(dim=-1)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.sum(dim=-1)
                - ref_rejected_logps.sum(dim=-1)
            ).detach()
        )

        rewards = chosen_rewards + rejected_rewards
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margin = chosen_rewards - rejected_rewards

        return {
            "{split}/loss": loss.mean(),
            "{split}/rewards_chosen": chosen_rewards.mean(),
            "{split}/rewards_rejected": rejected_rewards.mean(),
            "{split}/rewards": rewards.mean(),
            "{split}/rewards_accuracy": reward_accuracies.mean(),
            "{split}/rewards_margin": reward_margin.mean(),
            "{split}/logps_chosen": policy_chosen_logps.sum(dim=-1).mean(),
            "{split}/logps_rejected": policy_rejected_logps.sum(dim=-1).mean(),
        }

    def train_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        metrics = self.loss(
            chosen_input_ids=batch["chosen_input_ids"],
            chosen_attention_mask=batch["chosen_attention_mask"],
            chosen_labels=batch["chosen_labels"],
            rejected_input_ids=batch["rejected_input_ids"],
            rejected_attention_mask=batch["rejected_attention_mask"],
            rejected_labels=batch["rejected_labels"],
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
            chosen_input_ids=batch["chosen_input_ids"],
            chosen_attention_mask=batch["chosen_attention_mask"],
            chosen_labels=batch["chosen_labels"],
            rejected_input_ids=batch["rejected_input_ids"],
            rejected_attention_mask=batch["rejected_attention_mask"],
            rejected_labels=batch["rejected_labels"],
        )
        return self.gather_metrics(metrics)
