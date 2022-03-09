from typing import List, Optional

from nucleus.pydantic_base import DictCompatibleImmutableModel


class ScenesListEntry(DictCompatibleImmutableModel):
    """/dataset/{dataset_id}/scenes_list nested payload"""

    id: str
    reference_id: str
    type: str
    metadata: Optional[dict] = {}


class ScenesList(DictCompatibleImmutableModel):
    """/dataset/{dataset_id}/scenes_list payload"""

    scenes: List[ScenesListEntry]
