from pydantic import BaseModel  # pylint: disable=no-name-in-module


class ImmutableModel(BaseModel):
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
