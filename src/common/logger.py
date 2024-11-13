from __future__ import annotations

import atexit
import os
import sys
from typing import Any, ClassVar, Literal, Optional, TextIO

import wandb
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from common.utils import is_main_process, main_process_only


class Logger:
    _instance: ClassVar[Logger] = None
    wandb: Optional[wandb.sdk.wandb_run.Run] = None

    def __new__(
        cls,
        log_type: Literal["none", "wandb"] = "none",
        log_dir: Optional[os.PathLike] = None,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        config: dict[str, Any] = {},
    ):
        assert log_type in (
            "none",
            "wandb",
        ), f"log_type should be one of [wandb, none], but got {log_type}"

        if cls._instance is None:
            self = cls._instance = super().__new__(cls)

            self.log_type = log_type
            self.log_dir = log_dir
            self.project_name = project_name
            self.run_name = run_name

            if is_main_process():
                if self.log_type == "wandb":
                    self.wandb = wandb.init(
                        project=project_name, name=run_name, dir=log_dir, config=config
                    )

            atexit.register(self.close)
        else:
            assert (
                log_dir is None and project_name is None and run_name is None
            ), "Logger has been initialized."
        return cls._instance

    @main_process_only
    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.log_type == "wandb":
            self.wandb.log(metrics, step=step)

    @main_process_only
    def close(self) -> None:
        if self.log_type == "wandb":
            self.wandb.finish()

    @staticmethod
    @tqdm.external_write_mode()
    @main_process_only
    def print(
        *values,
        sep: Optional[str] = None,
        end: Optional[str] = None,
        file: Optional[TextIO] = None,
        flush: bool = False,
    ):
        print(*values, sep=sep, end=end, file=file or sys.stdout, flush=flush)

    @staticmethod
    @tqdm.external_write_mode()
    @main_process_only
    def print_table(
        title: str, data: dict[str, list[Any]], max_num_rows: Optional[int] = None
    ):
        cols = list(data.keys())
        rows = list(zip(*data.values()))

        if max_num_rows is None:
            max_num_rows = len(rows)

        rows = [list(map(str, row)) for row in rows]

        table = Table(title=title, show_lines=True, title_justify="left")
        for col in cols:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        Console(soft_wrap=True, markup=False, emoji=False).print(table)
