from enum import Enum
from typing import Dict, Optional, Any, Union, List
from .constants import (
    DATASET_ITEM_ID_KEY,
    REFERENCE_ID_KEY,
    METADATA_KEY,
    X_KEY,
    Y_KEY,
    WIDTH_KEY,
    HEIGHT_KEY,
    GEOMETRY_KEY,
    BOX_TYPE,
    POLYGON_TYPE,
    LABEL_KEY,
    TYPE_KEY,
    VERTICES_KEY,
)

class Segment:
    def __init__(self, label: str, index: int, metadata: Optional[dict] = None):
        self.label = label
        self.index = index
        self.metadata = metadata

    def __str__(self):
        return self.to_payload()

    @classmethod
    def from_json(cls, payload: dict):
        return cls(payload.get(LABEL_KEY), payload.get(INDEX_KEY), payload.get(METADATA_KEY, None))

    def to_payload(self) -> dict:
        payload = {
            LABEL_KEY: self.label,
            INDEX_KEY: self.index,
        }
        if self.metadata is not None:
            payload[METADATA_KEY] = self.metadata
        return payload

class SegmentationAnnotation:
    def __init__(self, mask_url: str, annotations: List[Segment], reference_id: Optional[str] = None, item_id: Optional[str] = None):
        if bool(reference_id) == bool(item_id):
            raise Exception(
                "You must specify either a reference_id or an item_id for an annotation."
            )
        self.mask_url = mask_url
        self.annotations = annotations
        self.reference_id = reference_id
        self.item_id = item_id


    def __str__(self):

    @classmethod
    def from_json(cls, payload: dict):

    def to_payload(self) -> dict:
        payload = {
            MASK_URL_KEY: self.mask_url,
            ANNOTATIONS_KEY: [ann.to_payload for ann in self.annotations],
        }


class AnnotationTypes(Enum):
    BOX = BOX_TYPE
    POLYGON = POLYGON_TYPE


# TODO: Add base annotation class to reduce repeated code here
class BoxAnnotation:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        label: str,
        x: Union[float, int],
        y: Union[float, int],
        width: Union[float, int],
        height: Union[float, int],
        reference_id: str = None,
        item_id: str = None,
        metadata: Optional[Dict] = None,
    ):
        if bool(reference_id) == bool(item_id):
            raise Exception(
                "You must specify either a reference_id or an item_id for an annotation."
            )
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.reference_id = reference_id
        self.item_id = item_id
        self.metadata = metadata if metadata else {}

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            x=geometry.get(X_KEY, 0),
            y=geometry.get(Y_KEY, 0),
            width=geometry.get(WIDTH_KEY, 0),
            height=geometry.get(HEIGHT_KEY, 0),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        return {
            LABEL_KEY: self.label,
            TYPE_KEY: BOX_TYPE,
            GEOMETRY_KEY: {
                X_KEY: self.x,
                Y_KEY: self.y,
                WIDTH_KEY: self.width,
                HEIGHT_KEY: self.height,
            },
            REFERENCE_ID_KEY: self.reference_id,
            METADATA_KEY: self.metadata,
        }

    def __str__(self):
        return str(self.to_payload())


# TODO: Add Generic type for 2D point
class PolygonAnnotation:
    def __init__(
        self,
        label: str,
        vertices: List[Any],
        reference_id: str = None,
        item_id: str = None,
        metadata: Optional[Dict] = None,
    ):
        if bool(reference_id) == bool(item_id):
            raise Exception(
                "You must specify either a reference_id or an item_id for an annotation."
            )
        self.label = label
        self.vertices = vertices
        self.reference_id = reference_id
        self.item_id = item_id
        self.metadata = metadata if metadata else {}

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            vertices=geometry.get(VERTICES_KEY, []),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        return {
            LABEL_KEY: self.label,
            TYPE_KEY: POLYGON_TYPE,
            GEOMETRY_KEY: {VERTICES_KEY: self.vertices},
            REFERENCE_ID_KEY: self.reference_id,
            METADATA_KEY: self.metadata,
        }

    def __str__(self):
        return str(self.to_payload())
