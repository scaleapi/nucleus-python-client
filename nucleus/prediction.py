"""
All of the prediction types supported. In general, prediction types are the same
as annotation types, but come with additional, optional data that can be attached
such as confidence or probability distributions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

from .annotation import (
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    Keypoint,
    KeypointsAnnotation,
    LineAnnotation,
    Point,
    Point3D,
    PolygonAnnotation,
    Segment,
    SegmentationAnnotation,
)
from .constants import (
    ANNOTATION_ID_KEY,
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CATEGORY_TYPE,
    CLASS_PDF_KEY,
    CONFIDENCE_KEY,
    CUBOID_TYPE,
    DIMENSIONS_KEY,
    EMBEDDING_VECTOR_KEY,
    GEOMETRY_KEY,
    HEIGHT_KEY,
    KEYPOINTS_KEY,
    KEYPOINTS_NAMES_KEY,
    KEYPOINTS_SKELETON_KEY,
    KEYPOINTS_TYPE,
    LABEL_KEY,
    LINE_TYPE,
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
    Y_KEY,
    YAW_KEY,
)


def from_json(payload: dict):
    """Instantiates prediction object from schematized JSON dict payload."""
    type_key_to_type: Dict[str, Type[Prediction]] = {
        BOX_TYPE: BoxPrediction,
        LINE_TYPE: LinePrediction,
        POLYGON_TYPE: PolygonPrediction,
        KEYPOINTS_TYPE: KeypointsPrediction,
        CUBOID_TYPE: CuboidPrediction,
        CATEGORY_TYPE: CategoryPrediction,
    }
    type_key = payload.get(TYPE_KEY, None)
    PredictionCls = type_key_to_type.get(type_key, SegmentationPrediction)
    return PredictionCls.from_json(payload)


class SegmentationPrediction(SegmentationAnnotation):
    """Predicted segmentation mask on a 2D image.

    ::

        from nucleus import SegmentationPrediction

        segmentation = SegmentationPrediction(
            mask_url="s3://your-bucket-name/pred-seg-masks/image_2_pred_mask_id1.png",
            annotations=[
                Segment(label="grass", index="1"),
                Segment(label="road", index="2"),
                Segment(label="bus", index="3", metadata={"vehicle_color": "yellow"}),
                Segment(label="tree", index="4")
            ],
            reference_id="image_2",
            annotation_id="image_2_pred_mask_1",
        )

    Parameters:
        mask_url (str): A URL pointing to the segmentation prediction mask which is
          accessible to Scale. This URL can be a path to a local file.
          The mask is an HxW int8 array saved in PNG format,
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
        annotation_id (Optional[str]): For segmentation predictions, this value is ignored
          because there can only be one segmentation prediction per dataset item.
          Therefore regardless of annotation ID, if there is an existing
          segmentation on a dataset item, it will be ignored unless update=True
          is passed to :meth:`Dataset.annotate`, in which case it will be overwritten.
          Storing a custom ID here may be useful in order to tie this annotation
          to an external database, and its value will be returned for any export.
    """

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
            # metadata=payload.get(METADATA_KEY, None),  # TODO(sc: 422637)
        )


class BoxPrediction(BoxAnnotation):
    """Prediction of a bounding box.

    Parameters:
        label (str): The label for this annotation (e.g. car, pedestrian, bicycle)
        x (Union[float, int]): The distance, in pixels, between the left border
            of the bounding box and the left border of the image.
        y (Union[float, int]): The distance, in pixels, between the top border
            of the bounding box and the top border of the image.
        width (Union[float, int]): The width in pixels of the annotation.
        height (Union[float, int]): The height in pixels of the annotation.
        reference_id (str): User-defined ID of the image to which to apply this
            annotation.
        confidence: 0-1 indicating the confidence of the prediction.
        annotation_id (Optional[str]): The annotation ID that uniquely
            identifies this annotation within its target dataset item. Upon ingest,
            a matching annotation id will be ignored by default, and updated if
            update=True for dataset.annotate.  If no annotation ID is passed, one
            will be automatically generated using the label, x, y, width, and
            height, so that you can make inserts idempotently and identical boxes
            will be ignored.
        metadata (Optional[Dict]): Arbitrary key/value dictionary of info to
            attach to this annotation.  Strings, floats and ints are supported best
            by querying and insights features within Nucleus. For more details see
            our `metadata guide <https://nucleus.scale.com/docs/upload-metadata>`_.
        class_pdf: An optional complete class probability distribution on this
            annotation. Each value should be between 0 and 1 (inclusive), and sum up to
            1 as a complete distribution. This can be useful for computing entropy to
            surface places where the model is most uncertain.
        embedding_vector (Optional[List]): Custom embedding vector for this object annotation.
            If any custom object embeddings have been uploaded previously to this dataset,
            this vector must match the dimensions of the previously ingested vectors.
    """

    def __init__(
        self,
        label: str,
        x: Union[float, int],
        y: Union[float, int],
        width: Union[float, int],
        height: Union[float, int],
        reference_id: str,
        confidence: Optional[float] = None,
        annotation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        class_pdf: Optional[Dict] = None,
        embedding_vector: Optional[list] = None,
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
            embedding_vector=embedding_vector,
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
            embedding_vector=payload.get(EMBEDDING_VECTOR_KEY, None),
        )


class LinePrediction(LineAnnotation):
    """Prediction of a line.

    Parameters:
        label (str): The label for this prediction (e.g. car, pedestrian, bicycle).
        vertices List[:class:`Point`]: The list of points making up the line.
        reference_id (str): User-defined ID of the image to which to apply this
            annotation.
        confidence: 0-1 indicating the confidence of the prediction.
        annotation_id (Optional[str]): The annotation ID that uniquely identifies
            this annotation within its target dataset item. Upon ingest, a matching
            annotation id will be ignored by default, and updated if update=True
            for dataset.annotate.
        metadata (Optional[Dict]): Arbitrary key/value dictionary of info to
            attach to this prediction.  Strings, floats and ints are supported best
            by querying and insights features within Nucleus. For more details see
            our `metadata guide <https://nucleus.scale.com/docs/upload-metadata>`_.
        class_pdf: An optional complete class probability distribution on this
            annotation. Each value should be between 0 and 1 (inclusive), and sum up to
            1 as a complete distribution. This can be useful for computing entropy to
            surface places where the model is most uncertain.
    """

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


class PolygonPrediction(PolygonAnnotation):
    """Prediction of a polygon.

    Parameters:
        label (str): The label for this annotation (e.g. car, pedestrian, bicycle).
        vertices List[:class:`Point`]: The list of points making up the polygon.
        reference_id (str): User-defined ID of the image to which to apply this
            annotation.
        confidence: 0-1 indicating the confidence of the prediction.
        annotation_id (Optional[str]): The annotation ID that uniquely identifies
            this annotation within its target dataset item. Upon ingest, a matching
            annotation id will be ignored by default, and updated if update=True
            for dataset.annotate.
        metadata (Optional[Dict]): Arbitrary key/value dictionary of info to
            attach to this annotation.  Strings, floats and ints are supported best
            by querying and insights features within Nucleus. For more details see
            our `metadata guide <https://nucleus.scale.com/docs/upload-metadata>`_.
        class_pdf: An optional complete class probability distribution on this
            annotation. Each value should be between 0 and 1 (inclusive), and sum up to
            1 as a complete distribution. This can be useful for computing entropy to
            surface places where the model is most uncertain.
        embedding_vector: Custom embedding vector for this object annotation.
            If any custom object embeddings have been uploaded previously to this dataset,
            this vector must match the dimensions of the previously ingested vectors.
    """

    def __init__(
        self,
        label: str,
        vertices: List[Point],
        reference_id: str,
        confidence: Optional[float] = None,
        annotation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        class_pdf: Optional[Dict] = None,
        embedding_vector: Optional[list] = None,
    ):
        super().__init__(
            label=label,
            vertices=vertices,
            reference_id=reference_id,
            annotation_id=annotation_id,
            metadata=metadata,
            embedding_vector=embedding_vector,
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
            embedding_vector=payload.get(EMBEDDING_VECTOR_KEY, None),
        )


class KeypointsPrediction(KeypointsAnnotation):
    """Prediction of keypoints.

    Parameters:
        label (str): The label for this annotation (e.g. car, pedestrian, bicycle).
        keypoints (List[:class:`Keypoint`]): The list of keypoints objects.
        names (List[str]): A list that corresponds to the names of each keypoint.
        skeleton (List[List[int]]): A list of 2-length lists indicating a beginning
            and ending index for each line segment in the skeleton of this keypoint label.
        reference_id (str): User-defined ID of the image to which to apply this
            annotation.
        confidence: 0-1 indicating the confidence of the prediction.
        annotation_id (Optional[str]): The annotation ID that uniquely identifies
            this annotation within its target dataset item. Upon ingest, a matching
            annotation id will be ignored by default, and updated if update=True
            for dataset.annotate.
        metadata (Optional[Dict]): Arbitrary key/value dictionary of info to
            attach to this annotation.  Strings, floats and ints are supported best
            by querying and insights features within Nucleus. For more details see
            our `metadata guide <https://nucleus.scale.com/docs/upload-metadata>`_.
        class_pdf: An optional complete class probability distribution on this
            annotation. Each value should be between 0 and 1 (inclusive), and sum up to
            1 as a complete distribution. This can be useful for computing entropy to
            surface places where the model is most uncertain.
    """

    def __init__(
        self,
        label: str,
        keypoints: List[Keypoint],
        names: List[str],
        skeleton: List[List[int]],
        reference_id: str,
        confidence: Optional[float] = None,
        annotation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        class_pdf: Optional[Dict] = None,
    ):
        super().__init__(
            label=label,
            keypoints=keypoints,
            names=names,
            skeleton=skeleton,
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
            keypoints=[
                Keypoint.from_json(_) for _ in geometry.get(KEYPOINTS_KEY, [])
            ],
            names=geometry[KEYPOINTS_NAMES_KEY],
            skeleton=geometry[KEYPOINTS_SKELETON_KEY],
            reference_id=payload[REFERENCE_ID_KEY],
            confidence=payload.get(CONFIDENCE_KEY, None),
            annotation_id=payload.get(ANNOTATION_ID_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
            class_pdf=payload.get(CLASS_PDF_KEY, None),
        )


class CuboidPrediction(CuboidAnnotation):
    """A prediction of 3D cuboid.

    Parameters:
        label (str): The label for this annotation (e.g. car, pedestrian, bicycle)
        position (:class:`Point3D`): The point at the center of the cuboid
        dimensions (:class:`Point3D`): The length (x), width (y), and height (z) of the cuboid
        yaw (float): The rotation, in radians, about the Z axis of the cuboid
        reference_id (str): User-defined ID of the image to which to apply this annotation.
        confidence: 0-1 indicating the confidence of the prediction.
        annotation_id (Optional[str]): The annotation ID that uniquely identifies this
          annotation within its target dataset item. Upon ingest, a matching
          annotation id will be ignored by default, and updated if update=True
          for dataset.annotate.
        metadata (Optional[str]): Arbitrary key/value dictionary of info to attach to this
          annotation.  Strings, floats and ints are supported best by querying
          and insights features within Nucleus. For more details see our `metadata
          guide <https://nucleus.scale.com/docs/upload-metadata>`_.
        class_pdf: An optional complete class probability distribution on this
            annotation. Each value should be between 0 and 1 (inclusive), and sum up to
            1 as a complete distribution. This can be useful for computing entropy to
            surface places where the model is most uncertain.
    """

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


class CategoryPrediction(CategoryAnnotation):
    """A prediction of a category.

    Parameters:
        label: The label for this annotation (e.g. car, pedestrian, bicycle).
        reference_id: The reference ID of the image you wish to apply this annotation to.
        taxonomy_name: The name of the taxonomy this annotation conforms to.
          See :meth:`Dataset.add_taxonomy`.
        confidence: 0-1 indicating the confidence of the prediction.
        class_pdf: An optional complete class probability distribution on this
            prediction. Each value should be between 0 and 1 (inclusive), and sum up to
            1 as a complete distribution. This can be useful for computing entropy to
            surface places where the model is most uncertain.
        metadata: Arbitrary key/value dictionary of info to attach to this annotation.
            Strings, floats and ints are supported best by querying and insights
            features within Nucleus. For more details see our `metadata guide
            <https://nucleus.scale.com/docs/upload-metadata>`_.
    """

    def __init__(
        self,
        label: str,
        reference_id: str,
        taxonomy_name: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
        class_pdf: Optional[Dict] = None,
    ):
        super().__init__(
            label=label,
            taxonomy_name=taxonomy_name,
            reference_id=reference_id,
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
        return cls(
            label=payload.get(LABEL_KEY, 0),
            taxonomy_name=payload.get(TAXONOMY_NAME_KEY, None),
            reference_id=payload[REFERENCE_ID_KEY],
            confidence=payload.get(CONFIDENCE_KEY, None),
            metadata=payload.get(METADATA_KEY, {}),
            class_pdf=payload.get(CLASS_PDF_KEY, None),
        )


Prediction = Union[
    BoxPrediction,
    LinePrediction,
    PolygonPrediction,
    KeypointsPrediction,
    CuboidPrediction,
    CategoryPrediction,
    SegmentationPrediction,
]


@dataclass
class PredictionList:
    """Wrapper class separating a list of predictions by type."""

    box_predictions: List[BoxPrediction] = field(default_factory=list)
    line_predictions: List[LinePrediction] = field(default_factory=list)
    polygon_predictions: List[PolygonPrediction] = field(default_factory=list)
    keypoints_predictions: List[KeypointsPrediction] = field(
        default_factory=list
    )
    cuboid_predictions: List[CuboidPrediction] = field(default_factory=list)
    category_predictions: List[CategoryPrediction] = field(
        default_factory=list
    )
    segmentation_predictions: List[SegmentationPrediction] = field(
        default_factory=list
    )

    def items(self):
        return self.__dict__.items()

    def add_predictions(self, predictions: List[Prediction]):
        for prediction in predictions:
            if isinstance(prediction, BoxPrediction):
                self.box_predictions.append(prediction)
            elif isinstance(prediction, LinePrediction):
                self.line_predictions.append(prediction)
            elif isinstance(prediction, PolygonPrediction):
                self.polygon_predictions.append(prediction)
            elif isinstance(prediction, KeypointsPrediction):
                self.keypoints_predictions.append(prediction)
            elif isinstance(prediction, CuboidPrediction):
                self.cuboid_predictions.append(prediction)
            elif isinstance(prediction, CategoryPrediction):
                self.category_predictions.append(prediction)
            else:
                assert isinstance(
                    prediction, SegmentationPrediction
                ), f"Unexpected prediction type: {type(prediction)}"
                self.segmentation_predictions.append(prediction)

    def __len__(self):
        return (
            len(self.box_predictions)
            + len(self.line_predictions)
            + len(self.polygon_predictions)
            + len(self.keypoints_predictions)
            + len(self.cuboid_predictions)
            + len(self.category_predictions)
            + len(self.segmentation_predictions)
        )
