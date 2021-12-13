from nucleus.pydantic_base import DictCompatibleModel


class DatasetSize(DictCompatibleModel):
    count: int
