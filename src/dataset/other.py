from dataset.base import BaseUnpairedDataset, BasePairedDataset


class UnpairedAnthropicHH(BaseUnpairedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_fn(self, example: dict[str, any]) -> dict[str, str]:
        context = [
            {
                "role": "user" if i["role"] == "human" else "assistant",
                "content": i["text"],
            }
            for i in example["context"]
        ]
        chosen = [{"role": "assistant", "content": example["chosen"]["text"]}]
        rejected = [{"role": "assistant", "content": example["rejected"]["text"]}]
        return {"chosen": context + chosen, "rejected": context + rejected}


class UnpairedTLDR(BaseUnpairedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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