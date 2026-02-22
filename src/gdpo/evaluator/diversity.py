import os
from typing import Any

import datasets
import openai
import torch
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from tqdm import tqdm

from gdpo.config import EvaluateConfig
from gdpo.evaluator import Evaluator


class DiversityEvaluator(Evaluator):
    model: SentenceTransformer | openai.Client

    def __init__(self, config: EvaluateConfig, accelerator: Accelerator) -> None:
        super().__init__(config, accelerator)

        self.use_openai = config.use_openai
        if self.use_openai:
            api_key = os.environ.get("OPENAI_API_KEY", None)
            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.model = openai.Client(api_key=api_key)
        else:
            self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.dataset = datasets.load_dataset(
            "json", data_files=config.output_path, split="train",
        )

    def evaluate_sample(self, sentences: list[str]) -> dict[str, Any]:
        if self.use_openai:
            embeddings: openai.types.CreateEmbeddingResponse = (
                self.model.embeddings.create(
                    input=sentences,
                )
            )
            embeddings = torch.tensor([e.embedding for e in embeddings.data])
        else:
            embeddings = self.model.encode(sentences, show_progress_bar=False)

        cos_sim = torch.mm(
            F.normalize(embeddings), F.normalize(embeddings).transpose(0, 1),
        )
        indices = torch.triu_indices(len(sentences), len(sentences), offset=1)
        diversity = 1 - cos_sim[indices[0], indices[1]].mean().item()
        return {"diversity": diversity}

    def evaluate(self) -> dict[str, Any]:
        diversities = []
        for example in (pbar := tqdm(self.dataset)):
            sentences = example["outputs"]
            diversities.append(self.evaluate_sample(sentences)["diversity"])

            stats = torch.tensor(diversities)
            mean_val = stats.mean().item()
            std_err = stats.std().item() / len(stats) ** 0.5
            pbar.set_description(f"{mean_val:.4f} ± {std_err:.4f}")

        diversities = torch.tensor(diversities)
        mean = diversities.mean().item()
        stde = diversities.std().item() / len(diversities) ** 0.5

        return {"mean": mean, "std_err": stde}
