from typing import Any, Literal

import torch
from transformers import PreTrainedModel

from algo.offline.dpo import DPOTrainer


class GDPOTrainer(DPOTrainer):
    ref_model: PreTrainedModel

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
            rejected_labels (torch.LongTensor): rejected labeles (offline trajectory)
            reduce (Literal["none", "mean", "sum"], optional): reduction in vocab dimension. Defaults to "sum".

        Returns:
            tuple[torch.FloatTensor]: tuple of (logits, logps, mask)
        """
        # concatenate in batch dim
        concat_input_ids = torch.cat((chosen_input_ids, rejected_input_ids), dim=0)
        concat_attention_mask = torch.cat(
            (chosen_attention_mask, rejected_attention_mask), dim=0
        )
        concat_labels = torch.cat((chosen_labels, rejected_labels), dim=0)
        # concatenated forward
        logits: torch.FloatTensor = model(
            input_ids=concat_input_ids, attention_mask=concat_attention_mask
        )["logits"]
        logps, mask = self.compute_token_logps(
            logits=logits,
            labels=concat_labels,
            slide_mask=True,
            temperature=self.config.algorithm.temperature,
        )
        # reduce logps
        if reduce == "mean":
            logps = logps.sum(dim=-1) / mask.sum(dim=-1)
        elif reduce == "sum":
            logps = logps.sum(dim=-1)
        return (logits, logps, mask)

    def loss(
        self,
        chosen_input_ids: torch.LongTensor,
        chosen_attention_mask: torch.LongTensor,
        chosen_labels: torch.LongTensor,
        rejected_input_ids: torch.LongTensor,
        rejected_attention_mask: torch.LongTensor,
        rejected_labels: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        (policy_logits, policy_logps, mask) = self._concat_forward(
            model=self.model,
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_attention_mask,
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_attention_mask=rejected_attention_mask,
            rejected_labels=rejected_labels,
            reduce="mean" if self.loss_type == "ipo" else "sum",
        )

        # log reward
        with torch.no_grad():
            (ref_logits, ref_logps, _) = self._concat_forward(
                model=self.ref_model,
                chosen_input_ids=chosen_input_ids,
                chosen_attention_mask=chosen_attention_mask,
                chosen_labels=chosen_labels,
                rejected_input_ids=rejected_input_ids,
                rejected_attention_mask=rejected_attention_mask,
                rejected_labels=rejected_labels,
                reduce="mean" if self.loss_type == "ipo" else "sum",
            )

            # base reward
            kl_div = policy_logps - ref_logps
            log_rewards = (
                ref_logps
                + (ref_logits / self.gamma).log_softmax(dim=-1)[
                    :, :-1, self.tokenizer.eos_token_id
                ]
            )

            chosen_rewards, rejected_rewards = log_rewards.chunk(chunks=2, dim=0)
            chosen_mask, rejected_mask = mask.chunk(chunks=2, dim=0)

            # 1 if chosen, small number otherwise
            # working in log-space
            scores = (
                torch.cat(
                    (
                        log_rewards.new_full(
                            (log_rewards.shape[0] // 2, 1), fill_value=0
                        ),
                        log_rewards.new_full(
                            (log_rewards.shape[0] // 2, 1), fill_value=-8
                        ),
                    )
                )
                * self.alpha
            )
            # adding to the last token
            for i in range(len(log_rewards)):
                last_index = mask[i].nonzero()[-1]
                log_rewards[i, last_index] += scores[i]

        # detailed balance loss
        eos_logps = policy_logits.log_softmax(dim=-1)[
            :, :-1, self.tokenizer.eos_token_id
        ]
        log_flows = log_rewards - eos_logps
        detailed_balance = log_flows[:, :-1] - log_flows[:, 1:] + policy_logps[:, :-1]
        detailed_balance = (detailed_balance * mask).pow(2).sum(dim=-1).mean()

        metrics = {
            "{split}/loss": detailed_balance,
            "{split}/kl": (kl_div * mask).sum(),
            "{split}/rewards": (log_rewards.sum(dim=-1) / mask.sum(dim=-1)).mean(),
            "{split}/chosen_rewards": (
                chosen_rewards.sum(dim=-1) / chosen_mask.sum(dim=-1)
            ).mean(),
            "{split}/rejected_rewards": (
                rejected_rewards.sum(dim=-1) / rejected_mask.sum(dim=-1)
            ).mean(),
        }
        return metrics

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

    def eval_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        metrics = self.loss(
            chosen_input_ids=batch["chosen_input_ids"],
            chosen_attention_mask=batch["chosen_attention_mask"],
            chosen_labels=batch["chosen_labels"],
            rejected_input_ids=batch["rejected_input_ids"],
            rejected_attention_mask=batch["rejected_attention_mask"],
            rejected_labels=batch["rejected_labels"],
        )
        return self.gather_metrics(metrics)
