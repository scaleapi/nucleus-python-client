from nucleus.pydantic_base import DictCompatibleModel


class DatasetDetails(DictCompatibleModel):
    """Pydantic model to parse DatasetDetails response from JSON"""

    id: str
    name: str
    is_public: bool
