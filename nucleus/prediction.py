from typing import Dict, Optional, List
from .annotation import (
    BoxAnnotation,
    Point,
    PolygonAnnotation,
    Segment,
    SegmentationAnnotation,
    CuboidAnnotation,
    Point3D,
)
from .constants import (
    ANNOTATION_ID_KEY,
    BOX_TYPE,
    CUBOID_TYPE,
    POLYGON_TYPE,
    REFERENCE_ID_KEY,
    METADATA_KEY,
    GEOMETRY_KEY,
    LABEL_KEY,
    TYPE_KEY,
    X_KEY,
    Y_KEY,
    WIDTH_KEY,
    HEIGHT_KEY,
    CLASS_PDF_KEY,
    CONFIDENCE_KEY,
    VERTICES_KEY,
    ANNOTATIONS_KEY,
    MASK_URL_KEY,
    POSITION_KEY,
    DIMENSIONS_KEY,
    YAW_KEY,
)


def from_json(payload: dict):
    if payload.get(TYPE_KEY, None) == BOX_TYPE:
        return BoxPrediction.from_json(payload)
    elif payload.get(TYPE_KEY, None) == POLYGON_TYPE:
        return PolygonPrediction.from_json(payload)
    elif payload.get(TYPE_KEY, None) == CUBOID_TYPE:
        return CuboidPrediction.from_json(payload)
    else:
        return SegmentationPrediction.from_json(payload)


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
            reference_id=payload[REFERENCE_ID_KEY],
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
        reference_id: str,
        confidence: Optional[float] = None,
        annotation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        class_pdf: Optional[Dict] = None,
    ):
        super().__init__(
            label=label,
            x=x,
            y=y,
            width=width,
            height=height,
            reference_id=reference_id,
            annotation_id=annotation_id,
            metadata=metadata,
        )
        self.confidence = confidence
        self.class_pdf = class_pdf

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.confidence is not None:
            payload[CONFIDENCE_KEY] = self.confidence
        if self.class_pdf is not None:
            payload[CLASS_PDF_KEY] = self.class_pdf

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
            reference_id=payload[REFERENCE_ID_KEY],
            confidence=payload.get(CONFIDENCE_KEY, None),
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
            class_pdf=payload.get(CLASS_PDF_KEY, None),
        )


class PolygonPrediction(PolygonAnnotation):
    def __init__(
        self,
        label: str,
        vertices: List[Point],
        reference_id: str,
        confidence: Optional[float] = None,
        annotation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        class_pdf: Optional[Dict] = None,
    ):
        super().__init__(
            label=label,
            vertices=vertices,
            reference_id=reference_id,
            annotation_id=annotation_id,
            metadata=metadata,
        )
        self.confidence = confidence
        self.class_pdf = class_pdf

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.confidence is not None:
            payload[CONFIDENCE_KEY] = self.confidence
        if self.class_pdf is not None:
            payload[CLASS_PDF_KEY] = self.class_pdf

        return payload

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            vertices=[
                Point.from_json(_) for _ in geometry.get(VERTICES_KEY, [])
            ],
            reference_id=payload[REFERENCE_ID_KEY],
            confidence=payload.get(CONFIDENCE_KEY, None),
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
            class_pdf=payload.get(CLASS_PDF_KEY, None),
        )


class CuboidPrediction(CuboidAnnotation):
    def __init__(
        self,
        label: str,
        position: Point3D,
        dimensions: Point3D,
        yaw: float,
        reference_id: str,
        confidence: Optional[float] = None,
        annotation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        class_pdf: Optional[Dict] = None,
    ):
        super().__init__(
            label=label,
            position=position,
            dimensions=dimensions,
            yaw=yaw,
            reference_id=reference_id,
            annotation_id=annotation_id,
            metadata=metadata,
        )
        self.confidence = confidence
        self.class_pdf = class_pdf

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.confidence is not None:
            payload[CONFIDENCE_KEY] = self.confidence
        if self.class_pdf is not None:
            payload[CLASS_PDF_KEY] = self.class_pdf

        return payload

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            position=Point3D.from_json(geometry.get(POSITION_KEY, {})),
            dimensions=Point3D.from_json(geometry.get(DIMENSIONS_KEY, {})),
            yaw=geometry.get(YAW_KEY, 0),
            reference_id=payload[REFERENCE_ID_KEY],
            confidence=payload.get(CONFIDENCE_KEY, None),
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
            class_pdf=payload.get(CLASS_PDF_KEY, None),
        )
