import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .constants import (
    ANNOTATION_ID_KEY,
    ANNOTATIONS_KEY,
    BOX_TYPE,
    DATASET_ITEM_ID_KEY,
    GEOMETRY_KEY,
    HEIGHT_KEY,
    INDEX_KEY,
    ITEM_ID_KEY,
    LABEL_KEY,
    MASK_URL_KEY,
    METADATA_KEY,
    POLYGON_TYPE,
    REFERENCE_ID_KEY,
    TYPE_KEY,
    VERTICES_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)


class Annotation:
    reference_id: Optional[str] = None
    item_id: Optional[str] = None

    def _check_ids(self):
        if bool(self.reference_id) == bool(self.item_id):
            raise Exception(
                "You must specify either a reference_id or an item_id for an annotation."
            )

    @classmethod
    def from_json(cls, payload: dict):
        if payload.get(TYPE_KEY, None) == BOX_TYPE:
            return BoxAnnotation.from_json(payload)
        elif payload.get(TYPE_KEY, None) == POLYGON_TYPE:
            return PolygonAnnotation.from_json(payload)
        else:
            return SegmentationAnnotation.from_json(payload)

    def to_payload(self):
        raise NotImplementedError(
            "For serialization, use a specific subclass (i.e. SegmentationAnnotation), "
            "not the base annotation class."
        )

    def to_json(self) -> str:
        return json.dumps(self.to_payload())


@dataclass
class Segment:
    label: str
    index: int
    metadata: Optional[dict] = None

    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            label=payload.get(LABEL_KEY, ""),
            index=payload.get(INDEX_KEY, None),
            metadata=payload.get(METADATA_KEY, None),
        )

    def to_payload(self) -> dict:
        payload = {
            LABEL_KEY: self.label,
            INDEX_KEY: self.index,
        }
        if self.metadata is not None:
            payload[METADATA_KEY] = self.metadata
        return payload


@dataclass
class SegmentationAnnotation(Annotation):
    mask_url: str
    annotations: List[Segment]
    annotation_id: Optional[str] = None
    reference_id: Optional[str] = None
    item_id: Optional[str] = None

    def __post_init__(self):
        if not self.mask_url:
            raise Exception("You must specify a mask_url.")
        self._check_ids()

    @classmethod
    def from_json(cls, payload: dict):
        if MASK_URL_KEY not in payload:
            raise ValueError(f"Missing {MASK_URL_KEY} in json")
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

    def to_payload(self) -> dict:
        payload = {
            MASK_URL_KEY: self.mask_url,
            ANNOTATIONS_KEY: [ann.to_payload() for ann in self.annotations],
            ANNOTATION_ID_KEY: self.annotation_id,
        }
        if self.reference_id:
            payload[REFERENCE_ID_KEY] = self.reference_id
        else:
            payload[ITEM_ID_KEY] = self.item_id
        return payload


class AnnotationTypes(Enum):
    BOX = BOX_TYPE
    POLYGON = POLYGON_TYPE


@dataclass  # pylint: disable=R0902
class BoxAnnotation(Annotation):  # pylint: disable=R0902
    label: str
    x: Union[float, int]
    y: Union[float, int]
    width: Union[float, int]
    height: Union[float, int]
    reference_id: Optional[str] = None
    item_id: Optional[str] = None
    annotation_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self._check_ids()
        self.metadata = self.metadata if self.metadata else {}

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
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
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
            ANNOTATION_ID_KEY: self.annotation_id,
            METADATA_KEY: self.metadata,
        }


# TODO: Add Generic type for 2D point
@dataclass
class PolygonAnnotation(Annotation):
    label: str
    vertices: List[Any]
    reference_id: Optional[str] = None
    item_id: Optional[str] = None
    annotation_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self._check_ids()
        self.metadata = self.metadata if self.metadata else {}

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            vertices=geometry.get(VERTICES_KEY, []),
            reference_id=payload.get(REFERENCE_ID_KEY, None),
            item_id=payload.get(DATASET_ITEM_ID_KEY, None),
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        return {
            LABEL_KEY: self.label,
            TYPE_KEY: POLYGON_TYPE,
            GEOMETRY_KEY: {VERTICES_KEY: self.vertices},
            REFERENCE_ID_KEY: self.reference_id,
            ANNOTATION_ID_KEY: self.annotation_id,
            METADATA_KEY: self.metadata,
        }
