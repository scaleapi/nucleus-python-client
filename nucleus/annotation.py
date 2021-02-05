from enum import Enum
from typing import Dict, Optional, Any, Union, List
from .constants import DATASET_ITEM_ID_KEY, REFERENCE_ID_KEY, METADATA_KEY


class AnnotationTypes(Enum):
    BOX = "box"
    POLYGON = "polygon"


class BoxAnnotation:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        label: str,
        x: Union[float, int],
        y: Union[float, int],
        width: Union[float, int],
        height: Union[float, int],
        reference_id: Optional[str] = None,
        item_id: Optional[str] = None,
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
        geometry = payload.get("geometry", {})
        return cls(
            label=payload.get("label", 0),
            x=geometry.get("x", 0),
            y=geometry.get("y", 0),
            width=geometry.get("width", 0),
            height=geometry.get("height", 0),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        return {
            "label": self.label,
            "type": "box",
            "geometry": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "reference_id": self.reference_id,
            "metadata": self.metadata,
        }

    def __str__(self):
        return str(self.to_payload())


class PolygonAnnotation:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        label: str,
        vertices: List[Any],
        reference_id: Optional[str] = None,
        item_id: Optional[str] = None,
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
        geometry = payload.get("geometry", {})
        return cls(
            label=payload.get("label", 0),
            vertices=geometry.get("vertices", []),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        return {
            "label": self.label,
            "type": "polygon",
            "geometry": {"vertices": self.vertices},
            "reference_id": self.reference_id,
            "metadata": self.metadata,
        }

    def __str__(self):
        return str(self.to_payload())
