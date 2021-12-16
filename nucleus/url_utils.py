import urllib.request
from functools import wraps


def sanitize_field(field):
    return urllib.request.quote(field.encode("UTF-8"), safe="")


def sanitize_string_args(function):
    """Helper decorator that ensures that all arguments passed are url-safe."""

    @wraps(function)
    def sanitized_function(*args, **kwargs):
        sanitized_args = []
        sanitized_kwargs = {}
        for arg in args:
            if isinstance(arg, str):
                arg = sanitize_field(arg)
            sanitized_args.append(arg)
        for key, value in kwargs.items():
            if isinstance(value, str):
                value = sanitize_field(value)
            sanitized_kwargs[key] = value
        return function(*sanitized_args, **sanitized_kwargs)

    return sanitized_function
