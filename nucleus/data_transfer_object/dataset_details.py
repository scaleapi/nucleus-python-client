from nucleus.data_transfer_object.dict_compatible_model import (
    DictCompatibleModel,
)


class DatasetDetails(DictCompatibleModel):
    """Pydantic model to parse DatasetDetails response from JSON"""

    id: str
    name: str
    is_public: bool
