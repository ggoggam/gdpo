import torch
from transformers import PreTrainedModel

from algo.offline import OfflineTrainer


class GDPOTrainer(OfflineTrainer):
    ref_model: PreTrainedModel

    def _concat_forward(
        self,
        model: torch.nn.Module,
        concat_input_ids: torch.LongTensor,
        concat_attention_mask: torch.BoolTensor,
        concat_labels: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        logits = model(
            input_ids=concat_input_ids,
            attention_mask=concat_attention_mask,
        )["logits"]
        return ...

    def loss(
        self,
        concat_input_ids: torch.LongTensor,
        concat_attention_mask: torch.BoolTensor,
        concat_labels: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        policy_logits, policy_logps, mask = self._concat_forward(
            self.model, concat_input_ids, concat_attention_mask, concat_labels
        )

        # log reward
        with torch.no_grad():
            ref_logits, ref_logps, _ = self._concat_forward(
                self.ref_model, concat_input_ids, concat_attention_mask, concat_labels
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
