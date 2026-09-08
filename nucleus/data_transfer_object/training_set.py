"""Response models for training sets."""

from typing import List

from nucleus.pydantic_base import DictCompatibleModel


class TrainingSetItemsPage(DictCompatibleModel):
    """One page of a training set's member dataset-item ids."""

    item_ids: List[str]
    total: int
