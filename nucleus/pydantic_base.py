"""
NOTE:
We started using pydantic during v1 and are kind of stuck with it now unless we write a compatibility layers.
As a library we want to support v1 and v2 such that we're not causing downstream problems for our users.
This means we have to do some import shenanigans to support both v1 and v2.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Backwards compatibility is even uglier with mypy
    from pydantic.v1 import BaseModel, Extra, ValidationError
else:
    try:
        # NOTE: we always use pydantic v1 but have to do these shenanigans to support both v1 and v2
        from pydantic.v1 import BaseModel  # pylint: disable=no-name-in-module
    except ImportError:
        from pydantic import BaseModel


class ImmutableModel(BaseModel):  # pylint: disable=used-before-assignment
    class Config:
        allow_mutation = False


class DictCompatibleModel(BaseModel):
    """Backwards compatible wrapper where we transform dictionaries into Pydantic Models

    Allows us to access model.key with model["key"].
    """

    def __getitem__(self, key):
        return getattr(self, key)


class DictCompatibleImmutableModel(ImmutableModel):
    """Backwards compatible wrapper where we transform dictionaries into Pydantic Models

    Allows us to access model.key with model["key"].
    """

    def __getitem__(self, key):
        return getattr(self, key)
