import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import tyro


@dataclass
class GenerationConfig:
    n_samples: int = 1
    top_k: int = 0
    top_p: float = 0.95
    temperature: float = 1.0
    max_new_tokens: int = 256
    min_new_tokens: int = 128
    max_tokens: int = 512


@dataclass
class ModelConfig:
    name: str = tyro.MISSING
    block: str = tyro.MISSING
    # checkpoint, if available
    checkpoint: Optional[str] = None


@dataclass
class SFTConfig:
    ...


@dataclass
class DPOConfig:
    loss_type: Literal["sigmoid", "ipo", "normcdf"] = "sigmoid"
    beta: float = 0.1
    label_smoothing: float = 0.0


@dataclass
class GDPOConfig:
    alpha: float = 0.1
    beta: float = 0.1
    gamma: float = 0.1
    temperature: float = 1.0


@dataclass
class CommonConfig:
    run_name: str = tyro.MISSING
    model: ModelConfig = field(
        default_factory=functools.partial(
            ModelConfig, "facebook/opt-125m", "OPTDecoderLayer"
        )
    )

    debug: bool = False
    project_name: str = "gdpo"
    seed: int = 0


@dataclass
class TrainConfig(CommonConfig):
    log_type: list[str] = field(default_factory=lambda: ["rich", "wandb"])
    log_dir: str = "."
    log_every: int = 1
    eval_every: int = 1000
    eval_first: bool = False

    task: Literal["hh", "tldr"] = "hh"
    checkpoint: Optional[str] = None

    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    dynamo_backend: Literal["no", "inductor"] = "inductor"
    algorithm: Union[SFTConfig, DPOConfig, GDPOConfig] = field(default_factory=SFTConfig)

    epochs: int = 1
    train_proportion: float = 1.0
    num_workers: int = 0
    pin_memory: bool = False
    # per-device batch sizes
    batch_size: int = 16
    eval_batch_size: int = 16
    # length
    max_prompt_length: int = 384
    max_length: int = 768
    # proportion of train set to use
    train_proportion: float = 1.0
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 10.0
    # optimizer / scheduler
    lr: float = 5e-6
    scheduler: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: Optional[float] = 0.03
    weight_decay: float = 0.01


@dataclass
class EvaluateConfig(CommonConfig):
    evaluator_model: str = tyro.MISSING
    use_openai: bool = False
