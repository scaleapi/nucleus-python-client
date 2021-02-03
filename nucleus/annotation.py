from typing import Dict, Optional
from .constants import DATASET_ITEM_ID_KEY, REFERENCE_ID_KEY, METADATA_KEY


class BoxAnnotation:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        label: str,
        x: int,
        y: int,
        width: int,
        height: int,
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
        geometry = payload.get("geometry", {})
        return cls(
            payload.get("label", 0),
            geometry.get("x", 0),
            geometry.get("y", 0),
            geometry.get("width", 0),
            geometry.get("height", 0),
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
