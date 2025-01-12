import abc
import atexit
from typing import Any, Callable, ClassVar

from accelerate import PartialState


def _is_main_process() -> bool:
    """Checks if current process is the main process

    Returns:
        bool: True if main process else False
    """
    return PartialState().is_main_process


def main_process_only(func: Callable[..., Any]):
    """Returns a function wrapper that runs
    the function only on the main process

    Args:
        func (Callable[..., Any]): a function to wrap

    Returns:
        wrapper: a function wrapper
    """

    def wrapper(*args, **kwargs) -> Any:
        if _is_main_process():
            return func(*args, **kwargs)
        return

    return wrapper


class BaseLogger(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the logger. Used for keyword arguments."""

    @property
    @abc.abstractmethod
    def writer(self):
        """Exposes the underlying logger."""

    @abc.abstractmethod
    def log_config(self, config: dict[str, any], **kwargs) -> None:
        """Logs hyperparameter configurations.

        Args:
            config (dict[str, any]): Hyperparameter configurations.
            **kwargs: Additional arguments.
        """

    @abc.abstractmethod
    def log(self, metrics: dict[str, any], step: int, **kwargs) -> None:
        """Logs metrics.

        Args:
            metrics (dict[str, any]): Metrics to log.
            step (int): Step number to associate the metrics with.
            **kwargs: Additional arguments.
        """

    @abc.abstractmethod
    def log_table(
        self, title: str, data: dict[str, list[any]], **kwargs
    ) -> None:
        """Logs a table of metrics.

        Args:
            title (str): The title of the table.
            data (dict[str, list[any]]): The data (column name to values) to log.
            **kwargs: Additional arguments.
        """

    def print(self, *args, **kwargs):
        """Prints to the console. Override if necessary."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Gracefully closes the logger."""


class MultipleLogger(BaseLogger):
    """A singleton class that aggregates multiple loggers."""

    _instance: ClassVar["MultipleLogger"] = None

    def __new__(cls, loggers: list[BaseLogger]):
        assert isinstance(loggers, list) and len(loggers) > 0

        if cls._instance is None:
            self = cls._instance = super().__new__(cls)
            self._loggers = loggers
            atexit.register(self.close)
        else:
            assert self is cls._instance, "MultipleLogger is a singleton class."
        return cls._instance

    def name(self) -> str:
        return "multiple"

    def writer(self):
        return {logger.name: logger.writer for logger in self._loggers}

    def log_config(self, config: dict[str, any], **kwargs):
        for logger in self._loggers:
            kwargs = kwargs.get(logger.name, {})
            logger.log_config(config, **kwargs)

    def log(self, metrics: dict[str, any], step: int, **kwargs):
        for logger in self._loggers:
            kwargs = kwargs.get(logger.name, {})
            logger.log(metrics, step=step, **kwargs)

    def log_table(self, title, data, **kwargs):
        for logger in self._loggers:
            kwargs = kwargs.get(logger.name, {})
            logger.log_table(title, data, **kwargs)

    def print(self, *args, **kwargs):
        for logger in self._loggers:
            logger.print(*args, **kwargs)

    def close(self):
        for logger in self._loggers:
            logger.close()
