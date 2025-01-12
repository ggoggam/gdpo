import abc
import copy
from typing import Union

import datasets
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers import (AutoModel, AutoTokenizer, PreTrainedModel,
                          get_scheduler)
from transformers.utils import is_flash_attn_2_available

from config import SFTConfig
from dataset import (PairedAnthropicHH, PairedTLDR, UnpairedAnthropicHH,
                     UnpairedTLDR)
from mixin.base import (TrainerMixinProtocol,
                        TrainerWithReferenceModelMixinProtocol)
from mixin.utilities import TorchDtypeMixin


class BaseInstantiateModelMixin(abc.ABC, TorchDtypeMixin):
    @abc.abstractmethod
    def _instantiate_model(
        self: TrainerMixinProtocol, model_cls: type[PreTrainedModel]
    ) -> None: ...

    def _instantiate_tokenizer(
        self: TrainerMixinProtocol, relax_template: bool = True
    ):
        """Instantiates tokenizer from a checkpoint.

        Args:
            config (TrainConfig): Training configuration
            relax_template (bool, optional):
                Relaxes chat template requirement. Defaults to True.
        """
        self.logger.print(">>> Tokenizer Initialization <<<")
        # may need to download tokenizer
        with self.accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # set template
        if tokenizer.chat_template is None:
            tokenizer.chat_template = open(
                "config/template/chatml.jinja"
            ).read()
        else:
            # some cases require relaxing the template
            if "[INST]" in tokenizer.chat_template and relax_template:
                tokenizer.chat_template = open(
                    "config/template/llama.jinja"
                ).read()
        self.tokenizer = tokenizer

    def _instantiate_optimizer(self: TrainerMixinProtocol):
        """Instantiates optimizer from a class. Must be done after dataset instantiation."""
        self.logger.print(">>> Optimizer Initialization <<<")
        # optimizer
        no_decay_names = list(
            map(
                lambda x: x[0].lower() in {"bias", "layernorm.weight"},
                self.model.named_parameters(),
            )
        )
        grouped_params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n.lower() not in no_decay_names
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n.lower() in no_decay_names
                ],
                "weight_decay": 0.0,
            },
        ]
        # deepspeed optimizer
        if self.accelerator.distributed_type == "DEEPSPEED":
            use_cpu_adam = (
                self.accelerator.deepspeed_config.get("zero_optimization", {})
                .get("offload_optimizer", {})
                .get("device", "none")
                != "none"
            )
            optimizer_cls = DeepSpeedCPUAdam if use_cpu_adam else FusedAdam
            optimizer = optimizer_cls(grouped_params, lr=self.config.lr)
        else:
            optimizer = torch.optim.AdamW(grouped_params, lr=self.config.lr)
        self.optimizer = optimizer

    def _instantiate_scheduler(self: TrainerMixinProtocol):
        """Instantiates scheduler from optimizer."""
        self.logger.print(">>> Scheduler Initialization <<<")
        scheduler = get_scheduler(
            name=self.config.scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        self.scheduler = scheduler


class InstantiateModelMixin(BaseInstantiateModelMixin):
    def _instantiate_model(
        self: TrainerMixinProtocol, model_cls: type[AutoModel]
    ) -> Union[torch.nn.Module, PreTrainedModel]:
        """Instantiates model from a checkpoint.

        Args:
            model_cls (type[AutoModel]): model class, e.g. AutoModelForCausalLM
        """
        self.logger.print(">>> Model Initialization <<<")
        kwargs = {
            "torch_dtype": self._get_torch_dtype(self.config.mixed_precision),
            "attn_implementation": "flash_attention_2"
            if is_flash_attn_2_available()
            else None,
        }
        # may need to download model
        with self.accelerator.main_process_first():
            model = model_cls.from_pretrained(
                self.config.model.name
                if self.config.model.checkpoint is None
                else self.config.model.checkpoint,
                **kwargs,
            )
        self.model = model


class InstantiateModelMixinWithReferenceModel(BaseInstantiateModelMixin):
    def _instantiate_model(
        self: TrainerWithReferenceModelMixinProtocol, model_cls: type[AutoModel]
    ) -> Union[torch.nn.Module, PreTrainedModel]:
        """Instantiates model from a checkpoint.

        Args:
            model_cls (type[AutoModel]): model class, e.g. AutoModelForCausalLM
        """
        self.logger.print(">>> Model Initialization <<<")
        kwargs = {
            "torch_dtype": self._get_torch_dtype(self.config.mixed_precision),
            "attn_implementation": "flash_attention_2"
            if is_flash_attn_2_available()
            else None,
        }
        # may need to download model
        with self.accelerator.main_process_first():
            model = model_cls.from_pretrained(
                self.config.model.name
                if self.config.model.checkpoint is None
                else self.config.model.checkpoint,
                **kwargs,
            )
        self.model = model
        self.reference_model = copy.deepcopy(model)


class InstantiateTrainerDatasetMixin:
    def _instantiate_dataset(
        self: TrainerMixinProtocol, is_train: bool
    ) -> torch.utils.data.Dataset:
        """Instantiates dataset from a class.

        Args:
            is_train (bool): whether to instantiate training or evaluation dataset

        Returns:
            torch.utils.data.Dataset: instantiated dataset
        """
        if self.config.task == "hh":
            dataset_cls = (
                UnpairedAnthropicHH
                if isinstance(self.config.algorithm, SFTConfig)
                else PairedAnthropicHH
            )
        elif self.config.task == "tldr":
            dataset_cls = (
                UnpairedTLDR
                if isinstance(self.config.algorithm, SFTConfig)
                else PairedTLDR
            )

        split = dataset_cls.train_split if is_train else dataset_cls.eval_split
        if is_train and self.config.train_proportion < 1.0:
            split = f"{dataset_cls.train_split}[:{int(self.config.train_proportion * 100)}%]"
        with self.accelerator.main_process_first():
            hf_dataset = datasets.load_dataset(
                dataset_cls.dataset_name, split=split
            )
            return dataset_cls(
                dataset=hf_dataset,
                tokenizer=self.tokenizer,
                max_prompt_length=self.config.max_prompt_length,
                max_length=self.config.max_length,
            )

    def _instantiate_dataloader(
        self: TrainerMixinProtocol,
    ) -> torch.utils.data.DataLoader:
        """Instantiates dataloader from a dataset."""
        train_dataset = self._instantiate_dataset(True)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        # technically distributed evaluation will not be accurate
        # but will do for now
        eval_dataset = self._instantiate_dataset(False)
        self.eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.config.eval_batch_size,
            collate_fn=eval_dataset.collate_fn,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
