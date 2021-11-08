"""
Adding ground truth to your dataset in Nucleus allows you to visualize annotations,
query dataset items based on the annotations they contain, and evaluate ModelRuns by
comparing predictions to ground truth.

Nucleus supports 2D bounding box, polygon, cuboid, and segmentation annotations.
Cuboid annotations can only be uploaded to a pointcloud DatasetItem.

When uploading an annotation, you need to specify which item you are annotating via
the reference_id you provided when uploading the image or pointcloud.

Ground truth uploads can be made idempotent by specifying an optional annotation_id for
each annotation. This id should be unique within the dataset_item so that
(reference_id, annotation_id) is unique within the dataset.

When uploading a mask annotation, Nucleus expects the mask file to be in PNG format
with each pixel being a 0-255 uint8. Currently, Nucleus only supports uploading masks
from URL.

Nucleus automatically enforces the constraint that each DatasetItem can have at most one
ground truth segmentation mask. As a consequence, if during upload a duplicate mask is
detected for a given image, by default it will be ignored. You can change this behavior
by specifying the optional 'update' flag. Setting update = true will replace the
existing segmentation with the new mask specified in the request body.

For ingesting large datasets, see the Guide for Large Ingestions.

.. todo::
    add link here
"""


import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union
from urllib.parse import urlparse

from .constants import (
    ANNOTATION_ID_KEY,
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CATEGORY_TYPE,
    MULTICATEGORY_TYPE,
    CUBOID_TYPE,
    DIMENSIONS_KEY,
    GEOMETRY_KEY,
    HEIGHT_KEY,
    INDEX_KEY,
    LABEL_KEY,
    LABELS_KEY,
    MASK_TYPE,
    MASK_URL_KEY,
    METADATA_KEY,
    POLYGON_TYPE,
    POSITION_KEY,
    REFERENCE_ID_KEY,
    TAXONOMY_NAME_KEY,
    TYPE_KEY,
    VERTICES_KEY,
    WIDTH_KEY,
    X_KEY,
    YAW_KEY,
    Y_KEY,
    Z_KEY,
)


class Annotation:
    """Simply a base class, not to be used directly

    Attributes:
        reference_id: The reference ID of the dataset item you wish to associate this
            annotation with
    """

    reference_id: str

    @classmethod
    def from_json(cls, payload: dict):
        if payload.get(TYPE_KEY, None) == BOX_TYPE:
            return BoxAnnotation.from_json(payload)
        elif payload.get(TYPE_KEY, None) == POLYGON_TYPE:
            return PolygonAnnotation.from_json(payload)
        elif payload.get(TYPE_KEY, None) == CUBOID_TYPE:
            return CuboidAnnotation.from_json(payload)
        elif payload.get(TYPE_KEY, None) == CATEGORY_TYPE:
            return CategoryAnnotation.from_json(payload)
        elif payload.get(TYPE_KEY, None) == MULTICATEGORY_TYPE:
            return MultiCategoryAnnotation.from_json(payload)
        else:
            return SegmentationAnnotation.from_json(payload)

    def to_payload(self):
        raise NotImplementedError(
            "For serialization, use a specific subclass (i.e. SegmentationAnnotation), "
            "not the base annotation class."
        )

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), allow_nan=False)


@dataclass  # pylint: disable=R0902
class BoxAnnotation(Annotation):  # pylint: disable=R0902
    """A bounding box annotation.

    Attributes:
        x: The distance, in pixels, between the left border of the bounding box and the
            left border of the image.
        y: The distance, in pixels, between the top border of the bounding box and the
            top border of the image.
        width: The width in pixels of the annotation.
        height: The height in pixels of the annotation.
        reference_id: The reference ID of the image you wish to apply this annotation to.
        annotation_id: The annotation ID that uniquely identifies this annotation within
            its target dataset item. Upon ingest, a matching annotation id will be
            ignored by default, and updated if update=True for dataset.annotate.
            If no annotation ID is passed, one will be automatically generated using the
            label, x, y, width, and height, so that you can make inserts idempotently
            and identical boxes will be ignored.
        label: The label for this annotation (e.g. car, pedestrian, bicycle)
        metadata: Arbitrary key/value dictionary of info to attach to this annotation.
            Strings, floats and ints are supported best by querying and insights
            features within Nucleus. For more details see our metadata guide.
            .. todo::
                Insert link to metadata guide
    """

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
    """A 2D point.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
    """

    x: float
    y: float

    @classmethod
    def from_json(cls, payload: Dict[str, float]):
        return cls(payload[X_KEY], payload[Y_KEY])

    def to_payload(self) -> dict:
        return {X_KEY: self.x, Y_KEY: self.y}


@dataclass
class PolygonAnnotation(Annotation):
    """A polygon annotation consisting of an ordered list of 2D points.

    Attributes:
        label: The label for this annotation (e.g. car, pedestrian, bicycle).
        vertices: The list of points making up the polygon.
        annotation_id: The annotation ID that uniquely identifies this annotation within
            its target dataset item. Upon ingest, a matching annotation id will be
            ignored by default, and updated if update=True for dataset.annotate.
        metadata: Arbitrary key/value dictionary of info to attach to this annotation.
            Strings, floats and ints are supported best by querying and insights
            features within Nucleus. For more details see our metadata guide.
            .. todo::
                Insert link to metadata guide
    """

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


@dataclass
class Point3D:
    """A point in 3D space.

    Attributes:
        x: The x coordinate of the point.
        y: The y coordinate of the point.
        z: The z coordinate of the point.
    """

    x: float
    y: float
    z: float

    @classmethod
    def from_json(cls, payload: Dict[str, float]):
        return cls(payload[X_KEY], payload[Y_KEY], payload[Z_KEY])

    def to_payload(self) -> dict:
        return {X_KEY: self.x, Y_KEY: self.y, Z_KEY: self.z}


@dataclass  # pylint: disable=R0902
class CuboidAnnotation(Annotation):  # pylint: disable=R0902
    """A 3D Cuboid annotation.

    Attributes:
        label: The label for this annotation (e.g. car, pedestrian, bicycle)
        position: The point at the center of the cuboid
        dimensions: The length (x), width (y), and height (z) of the cuboid
        yaw: The rotation, in radians, about the Z axis of the cuboid
        annotation_id: The annotation ID that uniquely identifies this annotation within
            its target dataset item. Upon ingest, a matching annotation id will be
            ignored by default, and updated if update=True for dataset.annotate.
        metadata: Arbitrary key/value dictionary of info to attach to this annotation.
            Strings, floats and ints are supported best by querying and insights
            features within Nucleus. For more details see our metadata guide.
            .. todo::
                Insert link to metadata guide
    """

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


@dataclass
class Segment:
    """Segment represents either a class or an instance depending on the task type.

    For semantic segmentation, this object should store the mapping between a single
    class index and the string label.

    For instance segmentation, you can use this class to store the label of a single
    instance, whose extent in the image is represented by the value of 'index'.

    In either case, additional metadata can be attached to the segment.

    Attributes:
        label: The label name of the class for the class or instance represented by index in the associated mask.
        index: The integer pixel value in the mask this mapping refers to.
        metadata: Arbitrary key/value dictionary of info to attach to this segment.
            Strings, floats and ints are supported best by querying and insights
            features within Nucleus. For more details see our metadata guide.
            .. todo::
                Insert link to metadata guide
    """

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
    """A segmentation mask on 2D image.

    Attributes:
        mask_url: A URL pointing to the segmentation prediction mask which is
            accessible to Scale. The mask is an HxW int8 array saved in PNG format,
            with each pixel value ranging from [0, N), where N is the number of possible
            classes (for semantic segmentation) or instances (for instance
            segmentation). The height and width of the mask must be the same as the
            original image. One example for semantic segmentation: the mask is 0 for
            pixels where there is background, 1 where there is a car, and 2 where there
            is a pedestrian. Another example for instance segmentation: the mask is 0
            for one car, 1 for another car, 2 for a motorcycle and 3 for another
            motorcycle. The class name for each value in the mask is stored in the list
            of Segment objects passed for "annotations"
        annotations: The list of mappings between the integer values contained in
            mask_url and string class labels. In the semantic segmentation example above
            these would map that 0 to background, 1 to car and 2 to pedestrian. In the
            instance segmentation example above, 0 and 1 would both be mapped to car,
            2 and 3 would both be mapped to motorcycle
        annotation_id: For segmentation annotations, this value is ignored because
          there can only be one segmentation annotation per dataset item. Therefore
          regardless of annotation ID, if there is an existing segmentation on a
          dataset item, it will be ignored unless update=True is passed to
          dataset.annotate, in which case it will be updated. Storing a custom ID here
          may be useful in order to tie this annotation to an external database, and
          its value will be returned for any export.
    """

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
    CATEGORY = CATEGORY_TYPE
    MULTICATEGORY = MULTICATEGORY_TYPE


@dataclass
class CategoryAnnotation(Annotation):
    """This class is not yet supported: Categorization support coming soon!"""

    label: str
    taxonomy_name: str
    reference_id: str
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self.metadata = self.metadata if self.metadata else {}

    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            label=payload[LABEL_KEY],
            taxonomy_name=payload[TAXONOMY_NAME_KEY],
            reference_id=payload[REFERENCE_ID_KEY],
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        return {
            LABEL_KEY: self.label,
            TAXONOMY_NAME_KEY: self.taxonomy_name,
            TYPE_KEY: CATEGORY_TYPE,
            GEOMETRY_KEY: {},
            REFERENCE_ID_KEY: self.reference_id,
            METADATA_KEY: self.metadata,
        }


@dataclass
class MultiCategoryAnnotation(Annotation):
    """This class is not yet supported: Categorization support coming soon!"""

    labels: List[str]
    taxonomy_name: str
    reference_id: str
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self.metadata = self.metadata if self.metadata else {}

    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            labels=payload[LABELS_KEY],
            taxonomy_name=payload[TAXONOMY_NAME_KEY],
            reference_id=payload[REFERENCE_ID_KEY],
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        return {
            LABELS_KEY: self.labels,
            TAXONOMY_NAME_KEY: self.taxonomy_name,
            TYPE_KEY: MULTICATEGORY_TYPE,
            GEOMETRY_KEY: {},
            REFERENCE_ID_KEY: self.reference_id,
            METADATA_KEY: self.metadata,
        }


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
