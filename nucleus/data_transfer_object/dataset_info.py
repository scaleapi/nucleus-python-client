from typing import List, Optional, Dict, Any


from nucleus.data_transfer_object.dict_compatible_model import (
    DictCompatibleModel,
)


class DatasetInfo(DictCompatibleModel):
    dataset_id: str
    name: str
    length: int
    model_run_ids: List[str]
    slice_ids: List[str]
    annotation_metadata_schema: Optional[Dict[str, Any]] = None
    item_metadata_schema: Optional[Dict[str, Any]] = None
