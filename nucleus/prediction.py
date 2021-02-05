from typing import Dict, Optional, List, Any
from .annotation import BoxAnnotation, PolygonAnnotation
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
        reference_id: Optional[str] = None,
        item_id: Optional[str] = None,
        confidence: Optional[float] = None,
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
            label=payload.get(LABEL_KEY, 0),
            x=geometry.get("x", 0),
            y=geometry.get("y", 0),
            width=geometry.get("width", 0),
            height=geometry.get("height", 0),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            confidence=payload.get(CONFIDENCE_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def __str__(self):
        return str(self.to_payload())


class PolygonPrediction(PolygonAnnotation):
    def __init__(
        self,
        label: str,
        vertices: List[Any],
        reference_id: Optional[str] = None,
        item_id: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(label, vertices, reference_id, item_id, metadata)
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
            label=payload.get(LABEL_KEY, 0),
            vertices=geometry.get("vertices", []),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            confidence=payload.get(CONFIDENCE_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def __str__(self):
        return str(self.to_payload())
