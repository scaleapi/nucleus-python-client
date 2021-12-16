import warnings
from functools import wraps
from typing import Callable


def deprecated(msg: str):
    """Adds a deprecation warning via the `warnings` lib which can be caught by linters.

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
            full_message = f"Calling {name} is deprecated: {msg}"
            # NOTE: stacklevel=2 makes sure that the level is applied to the decorated function
            warnings.warn(full_message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
