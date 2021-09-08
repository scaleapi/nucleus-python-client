import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union
from urllib.parse import urlparse

from .constants import (
    ANNOTATION_ID_KEY,
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CUBOID_TYPE,
    DIMENSIONS_KEY,
    GEOMETRY_KEY,
    HEIGHT_KEY,
    INDEX_KEY,
    LABEL_KEY,
    MASK_TYPE,
    MASK_URL_KEY,
    METADATA_KEY,
    POLYGON_TYPE,
    POSITION_KEY,
    REFERENCE_ID_KEY,
    TYPE_KEY,
    VERTICES_KEY,
    WIDTH_KEY,
    X_KEY,
    YAW_KEY,
    Y_KEY,
    Z_KEY,
)


class Annotation:
    reference_id: str

    @classmethod
    def from_json(cls, payload: dict):
        if payload.get(TYPE_KEY, None) == BOX_TYPE:
            return BoxAnnotation.from_json(payload)
        elif payload.get(TYPE_KEY, None) == POLYGON_TYPE:
            return PolygonAnnotation.from_json(payload)
        elif payload.get(TYPE_KEY, None) == CUBOID_TYPE:
            return CuboidAnnotation.from_json(payload)
        else:
            return SegmentationAnnotation.from_json(payload)

    def to_payload(self):
        raise NotImplementedError(
            "For serialization, use a specific subclass (i.e. SegmentationAnnotation), "
            "not the base annotation class."
        )

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), allow_nan=False)


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
    reference_id: str
    annotation_id: Optional[str] = None

    def __post_init__(self):
        if not self.mask_url:
            raise Exception("You must specify a mask_url.")

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
            reference_id=payload[REFERENCE_ID_KEY],
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
        )

    def to_payload(self) -> dict:
        payload = {
            TYPE_KEY: MASK_TYPE,
            MASK_URL_KEY: self.mask_url,
            ANNOTATIONS_KEY: [ann.to_payload() for ann in self.annotations],
            ANNOTATION_ID_KEY: self.annotation_id,
        }

        payload[REFERENCE_ID_KEY] = self.reference_id

        return payload


class AnnotationTypes(Enum):
    BOX = BOX_TYPE
    POLYGON = POLYGON_TYPE
    CUBOID = CUBOID_TYPE


@dataclass  # pylint: disable=R0902
class BoxAnnotation(Annotation):  # pylint: disable=R0902
    label: str
    x: Union[float, int]
    y: Union[float, int]
    width: Union[float, int]
    height: Union[float, int]
    reference_id: str
    annotation_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self.metadata = self.metadata if self.metadata else {}
        if self.annotation_id is None:
            self.annotation_id = f"{self.label}-{self.x}-{self.y}-{self.width}-{self.height}-{self.reference_id}"

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


@dataclass
class Point:
    x: float
    y: float

    @classmethod
    def from_json(cls, payload: Dict[str, float]):
        return cls(payload[X_KEY], payload[Y_KEY])

    def to_payload(self) -> dict:
        return {X_KEY: self.x, Y_KEY: self.y}


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    @classmethod
    def from_json(cls, payload: Dict[str, float]):
        return cls(payload[X_KEY], payload[Y_KEY], payload[Z_KEY])

    def to_payload(self) -> dict:
        return {X_KEY: self.x, Y_KEY: self.y, Z_KEY: self.z}


@dataclass
class PolygonAnnotation(Annotation):
    label: str
    vertices: List[Point]
    reference_id: str
    annotation_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self.metadata = self.metadata if self.metadata else {}
        if len(self.vertices) > 0:
            if not hasattr(self.vertices[0], X_KEY) or not hasattr(
                self.vertices[0], "to_payload"
            ):
                try:
                    self.vertices = [
                        Point(x=vertex[X_KEY], y=vertex[Y_KEY])
                        for vertex in self.vertices
                    ]
                except KeyError as ke:
                    raise ValueError(
                        "Use a point object to pass in vertices. For example, vertices=[nucleus.Point(x=1, y=2)]"
                    ) from ke

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            vertices=[
                Point.from_json(_) for _ in geometry.get(VERTICES_KEY, [])
            ],
            reference_id=payload[REFERENCE_ID_KEY],
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        payload = {
            LABEL_KEY: self.label,
            TYPE_KEY: POLYGON_TYPE,
            GEOMETRY_KEY: {
                VERTICES_KEY: [_.to_payload() for _ in self.vertices]
            },
            REFERENCE_ID_KEY: self.reference_id,
            ANNOTATION_ID_KEY: self.annotation_id,
            METADATA_KEY: self.metadata,
        }
        return payload


@dataclass  # pylint: disable=R0902
class CuboidAnnotation(Annotation):  # pylint: disable=R0902
    label: str
    position: Point3D
    dimensions: Point3D
    yaw: float
    reference_id: str
    annotation_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self.metadata = self.metadata if self.metadata else {}

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            position=Point3D.from_json(geometry.get(POSITION_KEY, {})),
            dimensions=Point3D.from_json(geometry.get(DIMENSIONS_KEY, {})),
            yaw=geometry.get(YAW_KEY, 0),
            reference_id=payload[REFERENCE_ID_KEY],
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        payload = {
            LABEL_KEY: self.label,
            TYPE_KEY: CUBOID_TYPE,
            GEOMETRY_KEY: {
                POSITION_KEY: self.position.to_payload(),
                DIMENSIONS_KEY: self.dimensions.to_payload(),
                YAW_KEY: self.yaw,
            },
        }
        payload[REFERENCE_ID_KEY] = self.reference_id
        if self.annotation_id:
            payload[ANNOTATION_ID_KEY] = self.annotation_id
        if self.metadata:
            payload[METADATA_KEY] = self.metadata

        return payload


def is_local_path(path: str) -> bool:
    return urlparse(path).scheme not in {"https", "http", "s3", "gs"}


def check_all_mask_paths_remote(
    annotations: Sequence[Union[Annotation]],
):
    for annotation in annotations:
        if hasattr(annotation, MASK_URL_KEY):
            if is_local_path(getattr(annotation, MASK_URL_KEY)):
                raise ValueError(
                    f"Found an annotation with a local path, which is not currently supported. Use a remote path instead. {annotation}"
                )
