import inspect
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    """
    Represents a ModelBundle.
    TODO fill this out with more than just a name potentially.
    """

    name: str
    func_or_class: Any = None
    code: str = ""

    def __str__(self):
        return f"ModelBundle(name={self.name})"


def create(name=None):
    def decorator(func_or_class):
        func_or_class._bundle_object = ModelBundle(
            name=name or func_or_class.__name__,
            func_or_class=func_or_class,
            code=inspect.getsource(func_or_class),
        )
        return func_or_class

    return decorator
