from datetime import timedelta
from functools import wraps
from typing import Any, Callable

import torch
from accelerate import PartialState


def is_main_process() -> bool:
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
        if is_main_process():
            return func(*args, **kwargs)
        return

    return wrapper


def gpu_timer(func: Callable[..., Any]):
    """Returns a function wrapper that times the function
    running on GPU for benchmarking

    Args:
        func (Callable[..., Any]): a function to time

    Returns:
        wrapper: a function wrapper
    """

    @wraps(func)
    @main_process_only
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            output = func(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()

            elapsed = start_event.elapsed_time(end_event)
            print(
                f"Elapsed GPU time of {func.__name__}: {timedelta(seconds=round(elapsed / 1000.0))}"
            )
        else:
            output = func(*args, **kwargs)

        return output

    return wrapper
