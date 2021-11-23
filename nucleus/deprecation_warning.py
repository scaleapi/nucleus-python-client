from functools import wraps
from typing import Callable

from nucleus import logger


def deprecated(msg: str):
    """Logs a deprecation warning to logger.warn.

    Args:
        msg: State reason of deprecation and point towards preferred practices

    Returns:
        Deprecation wrapped function
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # NOTE: __qualname looks a lot better for method calls
            name = (
                func.__qualname__
                if hasattr(func, "__qualname__")
                else func.__name__
            )
            logger.warn(f"Calling {name} is deprecated: {msg}")
            return func(*args, **kwargs)

        return wrapper

    return decorator