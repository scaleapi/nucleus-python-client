import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Type, Union
from urllib.parse import urlparse

from .constants import (
    ANNOTATION_ID_KEY,
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CATEGORY_TYPE,
    CUBOID_TYPE,
    DIMENSIONS_KEY,
    EMBEDDING_VECTOR_KEY,
    GEOMETRY_KEY,
    HEIGHT_KEY,
    INDEX_KEY,
    KEYPOINTS_KEY,
    KEYPOINTS_NAMES_KEY,
    KEYPOINTS_SKELETON_KEY,
    KEYPOINTS_TYPE,
    LABEL_KEY,
    LABELS_KEY,
    LINE_TYPE,
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
    VISIBLE_KEY,
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
        type_key_to_type: Dict[str, Type[Annotation]] = {
            BOX_TYPE: BoxAnnotation,
            LINE_TYPE: LineAnnotation,
            POLYGON_TYPE: PolygonAnnotation,
            KEYPOINTS_TYPE: KeypointsAnnotation,
            CUBOID_TYPE: CuboidAnnotation,
            CATEGORY_TYPE: CategoryAnnotation,
            MULTICATEGORY_TYPE: MultiCategoryAnnotation,
        }
        type_key = payload.get(TYPE_KEY, None)
        AnnotationCls = type_key_to_type.get(type_key, SegmentationAnnotation)
        return AnnotationCls.from_json(payload)

    def to_payload(self) -> dict:
        """Serializes annotation object to schematized JSON dict."""
        raise NotImplementedError(
            "For serialization, use a specific subclass (e.g. SegmentationAnnotation), "
            "not the base annotation class."
        )

    def to_json(self) -> str:
        """Serializes annotation object to schematized JSON string."""
        return json.dumps(self.to_payload(), allow_nan=False)

    def has_local_files_to_upload(self) -> bool:
        """Returns True if annotation has local files that need to be uploaded.

        Nearly all subclasses have no local files, so we default this to just return
        false. If the subclass has local files, it should override this method (but
        that is not the only thing required to get local upload of files to work.)
        """
        return False


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
            metadata={"vehicle_color": "red"},
            embedding_vector=[0.1423, 1.432, ...3.829],
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
        embedding_vector: Custom embedding vector for this object annotation.
            If any custom object embeddings have been uploaded previously to this dataset,
            this vector must match the dimensions of the previously ingested vectors.
    """

    label: str
    x: Union[float, int]
    y: Union[float, int]
    width: Union[float, int]
    height: Union[float, int]
    reference_id: str
    annotation_id: Optional[str] = None
    metadata: Optional[Dict] = None
    embedding_vector: Optional[list] = None

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
            embedding_vector=payload.get(EMBEDDING_VECTOR_KEY, None),
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
            EMBEDDING_VECTOR_KEY: self.embedding_vector,
        }

    def __eq__(self, other):
        return (
            self.label == other.label
            and self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
            and self.reference_id == other.reference_id
            and self.annotation_id == other.annotation_id
            and sorted(self.metadata.items()) == sorted(other.metadata.items())
            and self.embedding_vector == other.embedding_vector
        )


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
class LineAnnotation(Annotation):
    """A polyline annotation consisting of an ordered list of 2D points.
    A LineAnnotation differs from a PolygonAnnotation by not forming a closed
    loop, and by having zero area.

    ::

        from nucleus import LineAnnotation

        line = LineAnnotation(
            label="face",
            vertices=[Point(100, 100), Point(200, 300), Point(300, 200)],
            reference_id="person_image_1",
            annotation_id="person_image_1_line_1",
            metadata={"camera_mode": "portrait"},
        )

    Parameters:
        label (str): The label for this annotation.
        vertices (List[:class:`Point`]): The list of points making up the line.
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
            TYPE_KEY: LINE_TYPE,
            GEOMETRY_KEY: {
                VERTICES_KEY: [_.to_payload() for _ in self.vertices]
            },
            REFERENCE_ID_KEY: self.reference_id,
            ANNOTATION_ID_KEY: self.annotation_id,
            METADATA_KEY: self.metadata,
        }
        return payload


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
            metadata={"vehicle_color": "yellow"},
            embedding_vector=[0.1423, 1.432, ...3.829],
        )

    Parameters:
        label (str): The label for this annotation.
        vertices (List[:class:`Point`]): The list of points making up the polygon.
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
        embedding_vector: Custom embedding vector for this object annotation.
            If any custom object embeddings have been uploaded previously to this dataset,
            this vector must match the dimensions of the previously ingested vectors.
    """

    label: str
    vertices: List[Point]
    reference_id: str
    annotation_id: Optional[str] = None
    metadata: Optional[Dict] = None
    embedding_vector: Optional[list] = None

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
            embedding_vector=payload.get(EMBEDDING_VECTOR_KEY, None),
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
            EMBEDDING_VECTOR_KEY: self.embedding_vector,
        }
        return payload


@dataclass
class Keypoint:
    """A 2D point that has an additional visibility flag.

    Keypoints are intended to be part of a larger collection, and connected
    via a pre-defined skeleton. A keypoint in this skeleton may be visible
    or not-visible, and may be unlabeled and not visible. Because of this,
    the x, y coordinates may be optional, assuming that the keypoint is not
    visible, and would not be shown as part of the combined label.

    Parameters:
        x (Optional[float]): The x coordinate of the point.
        y (Optional[float]): The y coordinate of the point.
        visible (bool): The visibility of the point.
    """

    x: Optional[float] = None
    y: Optional[float] = None
    visible: bool = True

    def __post_init__(self):
        if self.visible and (self.x is None or self.y is None):
            raise ValueError(
                "Visible keypoints must have non-None x and y coordinates"
            )

    @classmethod
    def from_json(cls, payload: Dict[str, Union[float, bool]]):
        return cls(
            payload.get(X_KEY, None),
            payload.get(Y_KEY, None),
            bool(payload[VISIBLE_KEY]),
        )

    def to_payload(self) -> dict:
        return {
            X_KEY: self.x,
            Y_KEY: self.y,
            VISIBLE_KEY: self.visible,
        }


@dataclass
class KeypointsAnnotation(Annotation):
    """A keypoints annotation containing a list of keypoints and the structure
    of those keypoints: the naming of each point and the skeleton that connects
    those keypoints.

    ::

        from nucleus import KeypointsAnnotation

        keypoints = KeypointsAnnotation(
            label="face",
            keypoints=[Keypoint(100, 100), Keypoint(120, 120), Keypoint(visible=False), Keypoint(0, 0)],
            names=["point1", "point2", "point3", "point4"],
            skeleton=[[0, 1], [1, 2], [1, 3], [2, 3]],
            reference_id="image_2",
            annotation_id="image_2_face_keypoints_1",
            metadata={"face_direction": "forward"},
        )

    Parameters:
        label (str): The label for this annotation.
        keypoints (List[:class:`Keypoint`]): The list of keypoints objects.
        names (List[str]): A list that corresponds to the names of each keypoint.
        skeleton (List[List[int]]): A list of 2-length lists indicating a beginning and ending
            index for each line segment in the skeleton of this keypoint label.
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
    keypoints: List[Keypoint]
    names: List[str]
    skeleton: List[List[int]]
    reference_id: str
    annotation_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        self.metadata = self.metadata or {}
        if len(self.keypoints) != len(self.names):
            raise ValueError(
                "The list of keypoints must be the same length as the list of names"
            )
        if len(set(self.names)) != len(self.names):
            seen = set()
            for name in self.names:
                if name in seen:
                    raise ValueError(
                        f"The keypoint name '{name}' is repeated in the list of names"
                    )
                seen.add(name)

        max_segment_index = len(self.keypoints) - 1
        for segment in self.skeleton:
            if len(segment) != 2:
                raise ValueError(
                    "The keypoints skeleton must contain a list of line segments with exactly 2 indices"
                )
            for index in segment:
                if index > max_segment_index:
                    raise ValueError(
                        f"The skeleton index {index} is not a valid keypoint index"
                    )

    @classmethod
    def from_json(cls, payload: dict):
        geometry = payload.get(GEOMETRY_KEY, {})
        return cls(
            label=payload.get(LABEL_KEY, 0),
            keypoints=[
                Keypoint.from_json(_) for _ in geometry.get(KEYPOINTS_KEY, [])
            ],
            names=geometry[KEYPOINTS_NAMES_KEY],
            skeleton=geometry[KEYPOINTS_SKELETON_KEY],
            reference_id=payload[REFERENCE_ID_KEY],
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
        )

    def to_payload(self) -> dict:
        payload = {
            LABEL_KEY: self.label,
            TYPE_KEY: KEYPOINTS_TYPE,
            GEOMETRY_KEY: {
                KEYPOINTS_KEY: [_.to_payload() for _ in self.keypoints],
                KEYPOINTS_NAMES_KEY: self.names,
                KEYPOINTS_SKELETON_KEY: self.skeleton,
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
          accessible to Scale, or a local path. The mask is an HxW int8 array saved in PNG format,
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
    # metadata: Optional[dict] = None # TODO(sc: 422637)

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
            # metadata=payload.get(METADATA_KEY, None),  # TODO(sc: 422637)
        )

    def to_payload(self) -> dict:
        payload = {
            TYPE_KEY: MASK_TYPE,
            MASK_URL_KEY: self.mask_url,
            ANNOTATIONS_KEY: [ann.to_payload() for ann in self.annotations],
            ANNOTATION_ID_KEY: self.annotation_id,
            # METADATA_KEY: self.metadata,  # TODO(sc: 422637)
        }

        payload[REFERENCE_ID_KEY] = self.reference_id

        return payload

    def has_local_files_to_upload(self) -> bool:
        """Check if the mask url is local and needs to be uploaded."""
        if is_local_path(self.mask_url):
            if not os.path.isfile(self.mask_url):
                raise Exception(f"Mask file {self.mask_url} does not exist.")
            return True
        return False

    def __eq__(self, other):
        if not isinstance(other, SegmentationAnnotation):
            return False
        self.annotations = sorted(self.annotations, key=lambda x: x.index)
        other.annotations = sorted(other.annotations, key=lambda x: x.index)
        return (
            (self.annotation_id == other.annotation_id)
            and (self.annotations == other.annotations)
            and (self.mask_url == other.mask_url)
            and (self.reference_id == other.reference_id)
        )


class AnnotationTypes(Enum):
    BOX = BOX_TYPE
    LINE = LINE_TYPE
    POLYGON = POLYGON_TYPE
    KEYPOINTS = KEYPOINTS_TYPE
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


@dataclass
class AnnotationList:
    """Wrapper class separating a list of annotations by type."""

    box_annotations: List[BoxAnnotation] = field(default_factory=list)
    line_annotations: List[LineAnnotation] = field(default_factory=list)
    polygon_annotations: List[PolygonAnnotation] = field(default_factory=list)
    keypoints_annotations: List[KeypointsAnnotation] = field(
        default_factory=list
    )
    cuboid_annotations: List[CuboidAnnotation] = field(default_factory=list)
    category_annotations: List[CategoryAnnotation] = field(
        default_factory=list
    )
    multi_category_annotations: List[MultiCategoryAnnotation] = field(
        default_factory=list
    )
    segmentation_annotations: List[SegmentationAnnotation] = field(
        default_factory=list
    )

    def add_annotations(self, annotations: List[Annotation]):
        for annotation in annotations:
            assert isinstance(
                annotation, Annotation
            ), "Expected annotation to be of type 'Annotation"

            if isinstance(annotation, BoxAnnotation):
                self.box_annotations.append(annotation)
            elif isinstance(annotation, LineAnnotation):
                self.line_annotations.append(annotation)
            elif isinstance(annotation, PolygonAnnotation):
                self.polygon_annotations.append(annotation)
            elif isinstance(annotation, CuboidAnnotation):
                self.cuboid_annotations.append(annotation)
            elif isinstance(annotation, KeypointsAnnotation):
                self.keypoints_annotations.append(annotation)
            elif isinstance(annotation, CategoryAnnotation):
                self.category_annotations.append(annotation)
            elif isinstance(annotation, MultiCategoryAnnotation):
                self.multi_category_annotations.append(annotation)
            else:
                assert isinstance(
                    annotation, SegmentationAnnotation
                ), f"Unexpected annotation type: {type(annotation)}"
                self.segmentation_annotations.append(annotation)

    def items(self):
        return self.__dict__.items()

    def __len__(self):
        return (
            len(self.box_annotations)
            + len(self.line_annotations)
            + len(self.polygon_annotations)
            + len(self.keypoints_annotations)
            + len(self.cuboid_annotations)
            + len(self.category_annotations)
            + len(self.multi_category_annotations)
            + len(self.segmentation_annotations)
        )


def is_local_path(path: str) -> bool:
    return urlparse(path).scheme not in {"https", "http", "s3", "gs"}


def check_all_mask_paths_remote(
    annotations: Sequence[Annotation],
):
    for annotation in annotations:
        if hasattr(annotation, MASK_URL_KEY):
            if is_local_path(getattr(annotation, MASK_URL_KEY)):
                raise ValueError(
                    "Found an annotation with a local path, which is not currently"
                    f"supported for asynchronous upload. Use a remote path instead, or try synchronous upload. {annotation}"
                )
