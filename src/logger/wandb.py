import os

from logger.base import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(self, log_dir: os.PathLike, project_name: str, run_name: str):
        import wandb

        self._writer = wandb.init(
            project=project_name,
            name=run_name,
            dir=log_dir,
        )

    def name(self) -> str:
        return "wandb"

    def writer(self):
        return self._writer

    def log_config(self, config: dict[str, any], **kwargs):
        self._writer.config.update(config, allow_val_change=True)

    def log(self, metrics: dict[str, any], step: int, **kwargs):
        self._writer.log(metrics, step=step, **kwargs)

    def log_table(self, title: str, data: dict[str, any], **kwargs):
        import wandb

        columns = list(data.keys())
        data = zip(*data.values())
        table = {title: wandb.Table(columns=columns, data=data)}
        self.log(table, **kwargs)

    def close(self):
        self._writer.finish()
