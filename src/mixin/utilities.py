import gc
from typing import Union

import optree
import torch
from torch import distributed as dist


class TorchDtypeMixin:
    @staticmethod
    def _get_torch_dtype(mixed_precision: str) -> torch.dtype:
        """Returns torch dtype based on mixed precision setting.

        Args:
            mixed_precision (str): mixed precision setting

        Returns:
            torch.dtype: corresponding torch dtype
        """
        if mixed_precision == "fp16":
            return torch.float16
        if mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32


class GatherMetricsMixin:
    @staticmethod
    def gather(
        *args: torch.Tensor,
        reduce_op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Gathers torch.Tensor across processes to main process

        Args:
            reduce_op (dist.ReduceOp, optional):
                Reduce operation. Defaults to torch.distributed.ReduceOp.AVG.

        Returns:
            tuple[torch.Tensor, ...] | torch.Tensor:
                Gathered metrics moved to cpu
        """
        if dist.is_initialized():
            for metric in args:
                dist.all_reduce(metric, reduce_op)
        gathered = tuple(metric.cpu().item() for metric in args)
        return gathered if len(gathered) > 1 else gathered[0]

    @staticmethod
    def gather_metrics(
        metrics: dict[str, torch.Tensor],
        reduce_op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> dict[str, torch.Tensor]:
        """Gathers torch.Tensor across processes to main process

        Args:
            metrics (dict[str, torch.Tensor]): dictionary of metrics
            reduce_op (dist.ReduceOp, optional):
                reduce operation. Defaults to torch.distributed.ReduceOp.AVG.

        Returns:
            dict[str, torch.Tensor]: Gathered metrics moved to cpu
        """
        if dist.is_initialized():
            # reduce metrics across all processes then move to cpu
            for k in metrics:
                dist.all_reduce(metrics[k], reduce_op)
        metrics = optree.tree_map(lambda t: t.cpu().item(), metrics)
        return metrics


class ComputeMixin:
    @staticmethod
    def compute_token_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        slide_mask: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor]:
        """Computes token-wise log probability given logits and labels.

        Args:
            logits (torch.FloatTensor): logit outputs
            labels (torch.LongTensor): labels (sentence trajectory)
            slide_mask (bool, optional):
                slides mask to the left if true. Defaults to False.
            temperature (float, optional):
                probability temperature. Defaults to 1.0.

        Returns:
            tuple[torch.FloatTensor, torch.BoolTensor]:
                masked log probability and mask.
        """
        assert logits.shape[:-1] == labels.shape, (
            "Logits and labels must have same shape in the first two dimensions"
        )

        shifted_logps = (logits / temperature).log_softmax(dim=-1)[:, :-1, :]
        shifted_labels = labels[:, 1:].clone()
        mask = shifted_labels.not_equal(-100)
        shifted_labels[~mask] = 0

        logps = shifted_logps.gather(
            dim=-1, index=shifted_labels.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        if slide_mask:
            first_unmasked = torch.nonzero(mask.cumsum(dim=1) == 1)
            first_unmasked[:, -1] -= 1
            mask[first_unmasked[:, 0], first_unmasked[:, 1]] = True
        return logps * mask, mask


class ReleaseMemoryMixin:
    @staticmethod
    def release_cuda_memory(*objects):
        """Properly releases CUDA memory by moving tensors to CPU
        then freeing them by garbage collection.

        https://muellerzr.github.io/til/free_memory.html

        Args:
            objects: tensors or modules to release memory
        """
        if not isinstance(objects, list):
            objects = [objects]

        for i in range(len(objects)):
            if hasattr(objects[i], "to"):
                objects[i] = objects[i].to("cpu")
            objects[i] = None

        gc.collect()
        torch.cuda.empty_cache()
        return objects
