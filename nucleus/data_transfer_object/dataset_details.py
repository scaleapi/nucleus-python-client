from pydantic import BaseModel


class DatasetDetails(BaseModel):
    """Pydantic model to parse DatasetDetails response from JSON"""
    id: str
    name: str
    is_public: bool