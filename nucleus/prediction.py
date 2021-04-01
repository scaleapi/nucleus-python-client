from typing import Dict, Optional, List, Any
from .annotation import (
    BoxAnnotation,
    PolygonAnnotation,
    Segment,
    SegmentationAnnotation,
)
from .constants import (
    ANNOTATION_ID_KEY,
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
    ANNOTATIONS_KEY,
    ITEM_ID_KEY,
    MASK_URL_KEY,
)


class SegmentationPrediction(SegmentationAnnotation):
    # No need to define init or to_payload methods because
    # we default to functions defined in the parent class
    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            mask_url=payload[MASK_URL_KEY],
            annotations=[
                Segment.from_json(ann)
                for ann in payload.get(ANNOTATIONS_KEY, [])
            ],
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(ITEM_ID_KEY, None),
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
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
        annotation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(
            label,
            x,
            y,
            width,
            height,
            reference_id,
            item_id,
            annotation_id,
            metadata,
        )
        self.confidence = confidence

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.confidence is not None:
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
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
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
        annotation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(
            label, vertices, reference_id, item_id, annotation_id, metadata
        )
        self.confidence = confidence

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.confidence is not None:
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
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def __str__(self):
        return str(self.to_payload())
