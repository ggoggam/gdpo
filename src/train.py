import os
import warnings

import tyro
from accelerate import Accelerator
from accelerate.utils import (GradientAccumulationPlugin, is_bf16_available,
                              is_cuda_available, set_seed)

from algo.offline.dpo import DPOTrainer
from algo.offline.gdpo import GDPOTrainer
from algo.offline.sft import SFTTrainer
from config import DPOConfig, GDPOConfig, SFTConfig, TrainConfig

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(
        {
            "sft": (
                "Supervised Fine-tuning",
                TrainConfig(algorithm=SFTConfig()),
            ),
            "dpo": (
                "Direct Preference Optimization",
                TrainConfig(algorithm=DPOConfig()),
            ),
            "gdpo": (
                "GFlowNet Direct Preference Optimization",
                TrainConfig(algorithm=GDPOConfig()),
            ),
        }
    )

    # set seed
    set_seed(config.seed)

    # override the architecture block if FSDP is enabled
    if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = config.model.block

    # mixed precision override
    if config.mixed_precision == "bf16" and not is_bf16_available():
        warnings.warn(
            "bfloat16 is not available, using float32 instead. fp16 is not recommended for llm."
        )
        config.mixed_precision = "no"
    if config.mixed_precision == "fp16":
        warnings.warn(
            "float16 is not recommended for llm training. using float32 instead."
        )
        config.mixed_precision = "no"

    # compile override
    if config.dynamo_backend != "no" and not is_cuda_available():
        warnings.warn("dynamo is not available, using no backend instead.")
        config.dynamo_backend = "no"

    # accelerate
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        dynamo_backend=config.dynamo_backend,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.gradient_accumulation_steps,
        ),
    )

    # trainer
    trainer_cls = None
    if isinstance(config.algorithm, SFTConfig):
        trainer_cls = SFTTrainer
    elif isinstance(config.algorithm, DPOConfig):
        trainer_cls = DPOTrainer
    elif isinstance(config.algorithm, GDPOConfig):
        trainer_cls = GDPOTrainer
    trainer = trainer_cls(config=config, accelerator=accelerator)

    # train
    trainer.train()
