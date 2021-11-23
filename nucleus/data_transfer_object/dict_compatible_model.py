from pydantic import BaseModel


class DictCompatibleModel(BaseModel):
    """Backwards compatible wrapper where we transform dictionaries into Pydantic Models

    Allows us to access model.key with model["key"].
    """

    def __getitem__(self, key):
        return getattr(self, key)
