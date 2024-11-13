from dataclasses import dataclass
from typing import Literal, Optional


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
class CommonConfig:
    run_name: str
    project_name: str = "gdpo"

    debug: bool = False
    seed: int = 0


@dataclass
class TrainingConfig(CommonConfig):
    log_type: Literal["none", "wandb"] = "none"
    log_dir: str = "."
    log_every: int = 1
    eval_every: int = 1000

    epochs: int = 1
    # per-device batch sizes
    batch_size: int = 16
    eval_batch_size: int = 16
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
