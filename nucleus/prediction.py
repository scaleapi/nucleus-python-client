from typing import Dict, Optional, List, Any
from .annotation import BoxAnnotation, PolygonAnnotation
from .constants import (
    DATASET_ITEM_ID_KEY,
    REFERENCE_ID_KEY,
    METADATA_KEY,
    GEOMETRY_KEY,
    LABEL_KEY,
    X_KEY,
    Y_KEY,
    WIDTH_KEY,
    HEIGHT_KEY,
    CONFIDENCE_KEY,
    VERTICES_KEY,
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
            payload[CONFIDENCE_KEY] = self.confidence

        return payload

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
        reference_id: str = None,
        item_id: str = None,
        confidence: float = None,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(label, vertices, reference_id, item_id, metadata)
        self.confidence = confidence

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.confidence:
            payload[CONFIDENCE_KEY] = self.confidence

        return payload

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            vertices=geometry.get(VERTICES_KEY, []),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            confidence=payload.get(CONFIDENCE_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def __str__(self):
        return str(self.to_payload())
