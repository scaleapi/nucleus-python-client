from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from pydantic.v1 import validator
else:
    try:
        from pydantic.v1 import validator
    except ImportError:
        from pydantic import validator

from nucleus.pydantic_base import DictCompatibleModel


class DatasetInfo(DictCompatibleModel):
    """High-level :class:`Dataset` information

    Attributes:
        dataset_id: Nucleus-generated dataset ID
        name: User-defined name of dataset
        length: Number of :class:`DatasetItem` in :class:`Dataset`
        model_run_ids: (deprecated)
        slice_ids: List :class:`Slice` IDs associated with the :class:`Dataset`
        annotation_metadata_schema: Dict defining annotation-level metadata schema.
        item_metadata_schema: Dict defining item metadata schema.
        tags: List of tags associated with the :class:`Dataset`.
    """

    dataset_id: str
    name: str
    length: int
    model_run_ids: List[str]
    slice_ids: List[str]
    # TODO: Expand the following into pydantic models to formalize schema
    annotation_metadata_schema: Optional[Dict[str, Any]] = None
    item_metadata_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = []

    @validator(
        "tags", pre=True, always=True
    )  # pylint: disable=used-before-assignment
    def coerce_null_tags(cls, v):  # pylint: disable=no-self-argument
        return v if v is not None else []
