import json
from tqdm import tqdm
from rich import console, table
import torch

from .base import BaseLogger, main_process_only

class RichLogger(BaseLogger):
    @main_process_only
    def __init__(self):
        self._writer = console.Console(
            soft_wrap=True,
            markup=False,
            emoji=False,
        )

    def name(self) -> str:
        return "stdout"
    
    def writer(self):
        return self._writer
    
    @tqdm.external_write_mode()
    @main_process_only
    def log_config(self, config: dict[str, any], **kwargs) -> None:
        self._writer.print(">>> Configurations <<<")
        for key, value in config.items():
            self._writer.print(f"{key}: {value}")

    @tqdm.external_write_mode()
    @main_process_only
    def log(self, metrics: dict[str, any], step: int, **kwargs) -> None:
        self._writer.print(f"Step {step}:")
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            metrics[k] = v
        self._writer.print(json.dumps(metrics, indent=2))

    @tqdm.external_write_mode()
    @main_process_only
    def log_table(self, title: str, data: dict[str, any], **kwargs) -> None:
        rows = list(zip(*data.values()))
        max_rows = kwargs.get("max_rows", len(rows))
        tabular = table.Table(title=title, show_lines=True, title_justify="left")
        for col in data.keys():
            tabular.add_column(col)
        for row in rows[:max_rows]:
            tabular.add_row(*map(str, row))
        self._writer.print(tabular)

    @tqdm.external_write_mode()
    @main_process_only
    def print(self, *args, **kwargs):
        self._writer.print(*args, **kwargs)

    @main_process_only
    def close(self) -> None:
        pass
