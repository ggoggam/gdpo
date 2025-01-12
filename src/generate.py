import json
import time
from pathlib import Path

import datasets
import torch
import tyro
import vllm
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from config import CommonConfig
from dataset.other import PromptOnlyAnthropicHH, PromptOnlyTLDR
from mixin.instantiate import TorchDtypeMixin


class Generator(TorchDtypeMixin):
    model: vllm.LLM
    tokenizer: PreTrainedTokenizer

    def __init__(self, config: CommonConfig):
        self.config = config

        # model
        num_devices = torch.cuda.device_count()
        if num_devices % 2 == 1 and num_devices > 1:
            num_devices -= 1
        self.model = vllm.LLM(
            model=config.model.checkpoint,
            tokenizer=config.model.name,
            dtype=self._get_torch_dtype(config.mixed_precision),
            seed=config.seed,
            tensor_parallel_size=num_devices,
            gpu_memory_utilization=0.95,
        )

        # tokenizer
        tokenizer = self.model.get_tokenizer()
        tokenizer.chat_template = open("config/template/chatml.jinja").read()
        tokenizer.truncation_side = "left"
        self.tokenizer = tokenizer

        # dataset
        if config.task == "hh":
            dataset_cls = PromptOnlyAnthropicHH
            hf_dataset = datasets.load_dataset(
                dataset_cls.dataset_name, split=dataset_cls.eval_split
            ).map(dataset_cls.preprocess_fn)
        else:
            dataset_cls = PromptOnlyTLDR

            import polars

            # polars provide fast group-by
            temp_path = (
                Path(self.config.model.checkpoint) / "temp" / "dataset.parquet"
            )
            ds = datasets.load_dataset(
                dataset_cls.dataset_name, split=dataset_cls.eval_split
            )
            ds.to_parquet(temp_path)

            df = polars.scan_parquet(temp_path)
            df = (
                df.groupby("prompt")
                .agg([polars.col("chosen")], maintain_order=True)
                .collect()
            )

            hf_dataset = datasets.Dataset.from_list(df.to_dicts())
            hf_dataset = hf_dataset.map(dataset_cls.preprocess_fn)

        self.dataset = dataset_cls(
            dataset=hf_dataset,
            tokenizer=self.tokenizer,
            max_prompt_length=config.max_prompt_length,
            max_length=config.max_length,
        )

    def generate(self):
        path = Path(self.config.model.checkpoint) / "generated.jsonl"
        file = open(path, "w")

        for index in (
            pbar := tqdm(
                range(0, len(self.dataset), self.config.batch_size),
                desc="Generating...",
            )
        ):
            start, end = (
                index,
                min(index + self.config.batch_size, len(self.dataset)),
            )
            batch = self.dataset[start:end]

            encoded = self.tokenizer(
                text=batch["prompt"],
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
            )

            start_time = time.time()
            generated = self.model.generate(
                prompt_token_ids=encoded["input_ids"],
                sampling_params=vllm.SamplingParams(
                    n=config.n_samples,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=config.seed,
                    skip_special_tokens=True,
                    stop=[self.tokenizer.eos_token],
                ),
                use_tqdm=False,
            )
            pbar.set_description(f"Elapsed: {time.time() - start_time:.1f} sec")

            for history, prompt, references, outputs in zip(
                batch["history"], batch["prompt"], batch["chosen"], generated
            ):
                file.write(
                    json.dumps(
                        {
                            "history": history,
                            "prompt": prompt,
                            "outputs": [_.text for _ in outputs.outputs],
                            "references": references,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    config = tyro.cli(CommonConfig)

    assert config.model.checkpoint is not None, "Model checkpoint is required."

    generator = Generator(config)
    generator.generate()
