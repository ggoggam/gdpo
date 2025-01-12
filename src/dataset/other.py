from dataset.base import (
    BasePairedDataset,
    BasePromptOnlyDataset,
    BaseUnpairedDataset,
)


class PromptOnlyAnthropicHH(BasePromptOnlyDataset):
    dataset_name = "PKU-Alignment/processed-hh-rlhf"
    train_split = "train"
    eval_split = "test"

    def preprocess_fn(self, example: dict[str, any]) -> dict[str, str]:
        context = [
            {
                "role": "user" if i["role"] == "human" else "assistant",
                "content": i["text"],
            }
            for i in example["context"]
        ]
        return {
            "prompt": self.tokenizer.apply_chat_template(
                context, tokenize=False, add_generation_prompt=True
            ),
            "history": context,
            "chosen": [example["chosen"]["text"]],
        }


class UnpairedAnthropicHH(BaseUnpairedDataset):
    dataset_name = "PKU-Alignment/processed-hh-rlhf"
    train_split = "train"
    eval_split = "test"

    def preprocess_fn(self, example: dict[str, any]) -> dict[str, str]:
        context = [
            {
                "role": "user" if i["role"] == "human" else "assistant",
                "content": i["text"],
            }
            for i in example["context"]
        ]
        chosen = [{"role": "assistant", "content": example["chosen"]["text"]}]
        return {"chosen": context + chosen}


class PairedAnthropicHH(BasePairedDataset):
    dataset_name = "PKU-Alignment/processed-hh-rlhf"
    train_split = "train"
    eval_split = "test"

    def preprocess_fn(self, example: dict[str, any]) -> dict[str, str]:
        context = [
            {
                "role": "user" if i["role"] == "human" else "assistant",
                "content": i["text"],
            }
            for i in example["context"]
        ]
        chosen = [{"role": "assistant", "content": example["chosen"]["text"]}]
        rejected = [
            {"role": "assistant", "content": example["rejected"]["text"]}
        ]
        return {"chosen": context + chosen, "rejected": context + rejected}


class PromptOnlyTLDR(BasePromptOnlyDataset):
    dataset_name = "CarperAI/openai_summarize_comparisons"
    train_split = "train"
    eval_split = "test"

    def preprocess_fn(self, example: dict[str, any]) -> dict[str, str]:
        context = [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ]
        # TLDR has 1-N mapping between prompt and responses
        return {
            "prompt": self.tokenizer.apply_chat_template(
                context, tokenize=False, add_generation_prompt=True
            ),
            "history": context,
            "chosen": example["chosen"],
        }


class UnpairedTLDR(BaseUnpairedDataset):
    dataset_name = "CarperAI/openai_summarize_comparisons"
    train_split = "train"
    eval_split = "valid1"

    def preprocess_fn(self, example: dict[str, any]) -> dict[str, str]:
        context = [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ]
        chosen = [{"role": "assistant", "content": example["chosen"]}]
        return {"chosen": context + chosen}


class PairedTLDR(BasePairedDataset):
    dataset_name = "CarperAI/openai_summarize_comparisons"
    train_split = "train"
    eval_split = "valid1"

    def preprocess_fn(self, example: dict[str, any]) -> dict[str, str]:
        context = [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ]
        chosen = [{"role": "assistant", "content": example["chosen"]}]
        rejected = [{"role": "assistant", "content": example["rejected"]}]
        return {"chosen": context + chosen, "rejected": context + rejected}
