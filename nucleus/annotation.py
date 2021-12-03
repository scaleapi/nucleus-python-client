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
    MULTICATEGORY_TYPE,
    POLYGON_TYPE,
    POSITION_KEY,
    REFERENCE_ID_KEY,
    TAXONOMY_NAME_KEY,
    TYPE_KEY,
    VERTICES_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
    YAW_KEY,
    Z_KEY,
)


class Annotation:
    """Internal base class, not to be used directly.

    .. todo ::
        Inherit common constructor parameters from here
    """

    reference_id: str

    @classmethod
    def from_json(cls, payload: dict):
        """Instantiates annotation object from schematized JSON dict payload."""
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

    def to_payload(self) -> dict:
        """Serializes annotation object to schematized JSON dict."""
        raise NotImplementedError(
            "For serialization, use a specific subclass (e.g. SegmentationAnnotation), "
            "not the base annotation class."
        )

    def to_json(self) -> str:
        """Serializes annotation object to schematized JSON string."""
        return json.dumps(self.to_payload(), allow_nan=False)


@dataclass  # pylint: disable=R0902
class BoxAnnotation(Annotation):  # pylint: disable=R0902
    """A bounding box annotation.

    ::

        from nucleus import BoxAnnotation

        box = BoxAnnotation(
            label="car",
            x=0,
            y=0,
            width=10,
            height=10,
            reference_id="image_1",
            annotation_id="image_1_car_box_1",
            metadata={"vehicle_color": "red"}
        )

    Parameters:
        label (str): The label for this annotation.
        x (Union[float, int]): The distance, in pixels, between the left border
            of the bounding box and the left border of the image.
        y (Union[float, int]): The distance, in pixels, between the top border
            of the bounding box and the top border of the image.
        width (Union[float, int]): The width in pixels of the annotation.
        height (Union[float, int]): The height in pixels of the annotation.
        reference_id (str): User-defined ID of the image to which to apply this
            annotation.
        annotation_id (Optional[str]): The annotation ID that uniquely
            identifies this annotation within its target dataset item. Upon
            ingest, a matching annotation id will be ignored by default, and
            overwritten if update=True for dataset.annotate.  If no annotation
            ID is passed, one will be automatically generated using the label,
            x, y, width, and height, so that you can make inserts idempotently
            as identical boxes will be ignored.
        metadata (Optional[Dict]): Arbitrary key/value dictionary of info to
            attach to this annotation.  Strings, floats and ints are supported best
            by querying and insights features within Nucleus. For more details see
            our `metadata guide <https://nucleus.scale.com/docs/upload-metadata>`_.
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
    """A point in 2D space.

    Parameters:
        x (float): The x coordinate of the point.
        y (float): The y coordinate of the point.
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

    ::

        from nucleus import PolygonAnnotation

        polygon = PolygonAnnotation(
            label="bus",
            vertices=[Point(100, 100), Point(150, 200), Point(200, 100)],
            reference_id="image_2",
            annotation_id="image_2_bus_polygon_1",
            metadata={"vehicle_color": "yellow"}
        )

    Parameters:
        label (str): The label for this annotation.
        vertices List[:class:`Point`]: The list of points making up the polygon.
        reference_id (str): User-defined ID of the image to which to apply this
            annotation.
        annotation_id (Optional[str]): The annotation ID that uniquely identifies
            this annotation within its target dataset item. Upon ingest, a matching
            annotation id will be ignored by default, and updated if update=True
            for dataset.annotate.
        metadata (Optional[Dict]): Arbitrary key/value dictionary of info to
            attach to this annotation.  Strings, floats and ints are supported best
            by querying and insights features within Nucleus. For more details see
            our `metadata guide <https://nucleus.scale.com/docs/upload-metadata>`_.
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

    Parameters:
        x (float): The x coordinate of the point.
        y (float): The y coordinate of the point.
        z (float): The z coordinate of the point.
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

    ::

        from nucleus import CuboidAnnotation

        cuboid = CuboidAnnotation(
            label="car",
            position=Point3D(100, 100, 10),
            dimensions=Point3D(5, 10, 5),
            yaw=0,
            reference_id="pointcloud_1",
            annotation_id="pointcloud_1_car_cuboid_1",
            metadata={"vehicle_color": "green"}
        )

    Parameters:
        label (str): The label for this annotation.
        position (:class:`Point3D`): The point at the center of the cuboid
        dimensions (:class:`Point3D`): The length (x), width (y), and height (z) of the cuboid
        yaw (float): The rotation, in radians, about the Z axis of the cuboid
        reference_id (str): User-defined ID of the image to which to apply this annotation.
        annotation_id (Optional[str]): The annotation ID that uniquely identifies this
          annotation within its target dataset item. Upon ingest, a matching
          annotation id will be ignored by default, and updated if update=True
          for dataset.annotate.
        metadata (Optional[str]): Arbitrary key/value dictionary of info to attach to this
          annotation.  Strings, floats and ints are supported best by querying
          and insights features within Nucleus. For more details see our `metadata
          guide <https://nucleus.scale.com/docs/upload-metadata>`_.
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
    instance, whose extent in the image is represented by the value of ``index``.

    In both cases, additional metadata can be attached to the segment.

    Parameters:
        label (str): The label name of the class for the class or instance
          represented by index in the associated mask.
        index (int): The integer pixel value in the mask this mapping refers to.
        metadata (Optional[Dict]): Arbitrary key/value dictionary of info to attach to this segment.
          Strings, floats and ints are supported best by querying and insights
          features within Nucleus. For more details see our `metadata guide
          <https://nucleus.scale.com/docs/upload-metadata>`_.
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
    """A segmentation mask on a 2D image.

    When uploading a mask annotation, Nucleus expects the mask file to be in
    PNG format with each pixel being a 0-255 uint8. Currently, Nucleus only
    supports uploading masks from URL.

    Nucleus automatically enforces the constraint that each DatasetItem can
    have at most one ground truth segmentation mask. As a consequence, if
    during upload a duplicate mask is detected for a given image, by default it
    will be ignored. You can change this behavior by setting ``update = True``,
    which will replace the existing segmentation mask with the new mask.

    ::

        from nucleus import SegmentationAnnotation

        segmentation = SegmentationAnnotation(
            mask_url="s3://your-bucket-name/segmentation-masks/image_2_mask_id1.png",
            annotations=[
                Segment(label="grass", index="1"),
                Segment(label="road", index="2"),
                Segment(label="bus", index="3", metadata={"vehicle_color": "yellow"}),
                Segment(label="tree", index="4")
            ],
            reference_id="image_2",
            annotation_id="image_2_mask_1",
        )

    Parameters:
        mask_url (str): A URL pointing to the segmentation prediction mask which is
          accessible to Scale. The mask is an HxW int8 array saved in PNG format,
          with each pixel value ranging from [0, N), where N is the number of
          possible classes (for semantic segmentation) or instances (for instance
          segmentation).

          The height and width of the mask must be the same as the
          original image. One example for semantic segmentation: the mask is 0
          for pixels where there is background, 1 where there is a car, and 2
          where there is a pedestrian.

          Another example for instance segmentation: the mask is 0 for one car,
          1 for another car, 2 for a motorcycle and 3 for another motorcycle.
          The class name for each value in the mask is stored in the list of
          Segment objects passed for "annotations"
        annotations (List[:class:`Segment`]): The list of mappings between the integer values contained
          in mask_url and string class labels. In the semantic segmentation
          example above these would map that 0 to background, 1 to car and 2 to
          pedestrian. In the instance segmentation example above, 0 and 1 would
          both be mapped to car, 2 and 3 would both be mapped to motorcycle
        reference_id (str): User-defined ID of the image to which to apply this annotation.
        annotation_id (Optional[str]): For segmentation annotations, this value is ignored
          because there can only be one segmentation annotation per dataset item.
          Therefore regardless of annotation ID, if there is an existing
          segmentation on a dataset item, it will be ignored unless update=True
          is passed to :meth:`Dataset.annotate`, in which case it will be overwritten.
          Storing a custom ID here may be useful in order to tie this annotation
          to an external database, and its value will be returned for any export.
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
    """A category annotation.

    ::

        from nucleus import CategoryAnnotation

        category = CategoryAnnotation(
            label="dress",
            reference_id="image_1",
            taxonomy_name="clothing_type",
            metadata={"dress_color": "navy"}
        )

    Parameters:
        label (str): The label for this annotation.
        reference_id (str): User-defined ID of the image to which to apply this annotation.
        taxonomy_name (Optional[str]): The name of the taxonomy this annotation conforms to.
          See :meth:`Dataset.add_taxonomy`.
        metadata (Optional[Dict]): Arbitrary key/value dictionary of info to attach to this annotation.
          Strings, floats and ints are supported best by querying and insights
          features within Nucleus. For more details see our `metadata guide
          <https://nucleus.scale.com/docs/upload-metadata>`_.
    """

    label: str
    reference_id: str
    taxonomy_name: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self.metadata = self.metadata if self.metadata else {}

    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            label=payload[LABEL_KEY],
            reference_id=payload[REFERENCE_ID_KEY],
            taxonomy_name=payload.get(TAXONOMY_NAME_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        payload = {
            LABEL_KEY: self.label,
            TYPE_KEY: CATEGORY_TYPE,
            GEOMETRY_KEY: {},
            REFERENCE_ID_KEY: self.reference_id,
            METADATA_KEY: self.metadata,
        }
        if self.taxonomy_name is not None:
            payload[TAXONOMY_NAME_KEY] = self.taxonomy_name
        return payload


@dataclass
class MultiCategoryAnnotation(Annotation):
    """This class is not yet supported: MultiCategory annotation support coming soon!"""

    labels: List[str]
    reference_id: str
    taxonomy_name: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self.metadata = self.metadata if self.metadata else {}

    @classmethod
    def from_json(cls, payload: dict):
        return cls(
            labels=payload[LABELS_KEY],
            reference_id=payload[REFERENCE_ID_KEY],
            taxonomy_name=payload.get(TAXONOMY_NAME_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        payload = {
            LABELS_KEY: self.labels,
            TYPE_KEY: MULTICATEGORY_TYPE,
            GEOMETRY_KEY: {},
            REFERENCE_ID_KEY: self.reference_id,
            METADATA_KEY: self.metadata,
        }
        if self.taxonomy_name is not None:
            payload[TAXONOMY_NAME_KEY] = self.taxonomy_name
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
                    "Found an annotation with a local path, which is not currently"
                    f"supported. Use a remote path instead. {annotation}"
                )
