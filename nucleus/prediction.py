from typing import Dict, Optional
from .annotation import BoxAnnotation
from .constants import (
    DATASET_ITEM_ID_KEY,
    REFERENCE_ID_KEY,
    METADATA_KEY,
    CONFIDENCE_KEY,
    LABEL_KEY,
)


class BoxPrediction(BoxAnnotation):
    def __init__(
        self,
        label: str,
        x: int,
        y: int,
        width: int,
        height: int,
        reference_id: str = None,
        item_id: str = None,
        confidence: float = None,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(
            label, x, y, width, height, reference_id, item_id, metadata
        )
        self.confidence = confidence

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.confidence:
            payload["confidence"] = self.confidence

        return payload

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get("geometry", {})
        return cls(
            payload.get(LABEL_KEY, 0),
            geometry.get("x", 0),
            geometry.get("y", 0),
            geometry.get("width", 0),
            geometry.get("height", 0),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            confidence=payload.get(CONFIDENCE_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def __str__(self):
        return str(self.to_payload())
