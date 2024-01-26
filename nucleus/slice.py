import datetime
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import requests

from nucleus.annotation import Annotation
from nucleus.async_job import AsyncJob, EmbeddingsExportJob
from nucleus.constants import (
    EXPORT_FOR_TRAINING_KEY,
    EXPORTED_ROWS,
    ITEMS_KEY,
    SLICE_TAGS_KEY,
)
from nucleus.dataset_item import DatasetItem
from nucleus.errors import NucleusAPIError
from nucleus.prediction import Prediction
from nucleus.prediction import from_json as prediction_from_json
from nucleus.scene import Scene
from nucleus.utils import (
    KeyErrorDict,
    convert_export_payload,
    format_scale_task_info_response,
    paginate_generator,
)


class SliceBuilderMethods(str, Enum):
    """
    Which method to use for sampling the dataset items.
      - Random: randomly select items
      - Uniqueness: Prioritizes more unique images based on model embedding distance, so that the final sample has fewer similar images.
    """

    RANDOM = "Random"
    UNIQUENESS = "Uniqueness"

    def __contains__(self, item):
        try:
            self(item)
        except ValueError:
            return False
        return True

    @staticmethod
    def options():
        return list(map(lambda c: c.value, SliceBuilderMethods))


@dataclass
class SliceBuilderFilterAutotag:
    """
    Helper class for specifying an autotag filter for building a slice.

    Args:
        autotag_id: Filter items that belong to this autotag
        score_range: Specify the range of the autotag items' score that should be considered, between [-1, 1].
            For example, [-0.3, 0.7].
    """

    autotag_id: str
    score_range: List[int]

    def __post_init__(self):
        warn_msg = f"Autotag score range must be within [-1, 1]. But got {self.score_range}."
        assert len(self.score_range) == 2, warn_msg
        assert (
            min(self.score_range) >= -1 and max(self.score_range) <= 1
        ), warn_msg


@dataclass
class SliceBuilderFilters:
    """
    Optionally apply filters to the collection of dataset items when building the slice.
    Items can be filtered by an existing slice and/or an autotag.

    Args:
        slice_id: Build the slice from items pertaining to this slice
        autotag: Build the slice from items pertaining to an autotag (see SliceBuilderFilterAutotag)
    """

    slice_id: Optional[str] = None
    autotag: Optional[SliceBuilderFilterAutotag] = None


class SliceType(str, Enum):
    """
    Types of slices supported by Nucleus.
    """

    DATASET_ITEM = "dataset_item"
    OBJECT = "object"
    SCENE = "scene"

    def __contains__(self, item):
        try:
            self(item)
        except ValueError:
            return False
        return True

    @staticmethod
    def options():
        return list(map(lambda c: c.value, SliceType))


class Slice:
    """A Slice represents a subset of DatasetItems in your Dataset.

    Slices are subsets of your Dataset that unlock curation and exploration
    workflows.  Instead of thinking of your Datasets as collections of data, it
    is useful to think about them as a collection of Slices. For instance, your
    dataset may contain different weather scenarios, traffic conditions, or
    highway types.

    Perhaps your Models perform poorly on foggy weather scenarios; it is then
    useful to slice your dataset into a "foggy" slice, and fine-tune model
    performance on this slice until it reaches the performance you desire.

    Slices cannot be instantiated directly and instead must be created in the
    dashboard, or via API endpoint using :meth:`Dataset.create_slice`.

    ::

        import nucleus

        client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
        dataset = client.get_dataset("YOUR_DATASET_ID")

        ref_ids = ["interesting_item_1", "interesting_item_2"]
        slice = dataset.create_slice(name="interesting", reference_ids=ref_ids)
    """

    def __init__(self, slice_id: str, client):
        self.id = slice_id
        self._slice_id = slice_id
        self._client = client
        self._name = None
        self._dataset_id = None
        self._description = None
        self._created_at = None
        self._pending_job_count = None
        self._type = None

    def __repr__(self):
        return f"Slice(slice_id='{self.id}', name='{self.name}', dataset_id='{self.dataset_id}', description='{self.description}')"

    def __eq__(self, other):
        if self.id == other.id:
            if self._client == other._client:
                return True
        return False

    @property
    def created_at(self) -> Optional[datetime.datetime]:
        """Timestamp of creation of the slice

        Returns:
            datetime of creation or None if not created yet
        """
        if self._created_at is None:
            self._created_at = self.info().get("created_at", None)
        return self._created_at

    @property
    def pending_job_count(self) -> Optional[int]:
        if self._pending_job_count is None:
            self._pending_job_count = self.info().get(
                "pending_job_count", None
            )
        return self._pending_job_count

    @classmethod
    def from_request(cls, request, client):
        instance = cls(request["id"], client)
        instance._name = request.get("name", None)
        instance._dataset_id = request.get("dataset_id", None)
        instance._type = request.get("type", None)
        created_at_str = request.get("created_at").rstrip("Z")
        if hasattr(datetime.datetime, "fromisoformat"):
            instance._created_at = datetime.datetime.fromisoformat(
                created_at_str
            )
        else:
            fmt_str = r"%Y-%m-%dT%H:%M:%S.%f"  # replaces the fromisoformatm, not available in python 3.6
            instance._created_at = datetime.datetime.strptime(
                created_at_str, fmt_str
            )
        instance._pending_job_count = request.get("pending_job_count", None)
        return instance

    @property
    def slice_id(self):
        warnings.warn(
            "Using Slice.slice_id is deprecated. Prefer using Slice.id",
            DeprecationWarning,
        )
        return self._slice_id

    @property
    def name(self):
        """The name of the Slice."""
        if self._name is None:
            self._name = self.info()["name"]
        return self._name

    @property
    def description(self):
        """Slice description, plain text."""
        if self._description is None:
            self._description = self.info()["description"]
        return self._description

    @property
    def dataset_id(self):
        """The ID of the Dataset to which the Slice belongs."""
        if self._dataset_id is None:
            self._dataset_id = self.info()["dataset_id"]
        return self._dataset_id

    @property
    def tags(self):
        """Get all associated tags for this slice."""
        return self.info()["tags"]

    @property
    def type(self):
        """The type of the Slice."""
        if self._type is None:
            self._type = self.info()["type"]
        return self._type

    def items_generator(self, page_size=100000):
        """Generator yielding all dataset items in the dataset.

        ::

            collected_ref_ids = []
            for item in dataset.items_generator():
                print(f"Exporting item: {item.reference_id}")
                collected_ref_ids.append(item.reference_id)

        Args:
            page_size (int, optional): Number of items to return per page. If you are
                experiencing timeouts while using this generator, you can try lowering
                the page size.

        Yields:
            an iterable of DatasetItem objects.
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"slice/{self.id}/itemsPage",
            result_key=ITEMS_KEY,
            page_size=page_size,
        )
        for item_json in json_generator:
            yield DatasetItem.from_json(item_json)

    def dataset_items(self):
        """Fetch all DatasetItems contained in the Slice.

        We recommend using :meth:`Slice.items_generator` if the Slice has more than 200k items.

        Returns: list of DatasetItem objects

        """
        try:
            response = self._client.make_request(
                {}, f"slice/{self.id}", requests_command=requests.get
            )
        except NucleusAPIError as e:
            if e.status_code == 503:
                e.message += "/n Your request timed out while trying to get all the items in the slice. Please try slice.items_generator() instead."
            raise e

        dataset_item_jsons = response.get(ITEMS_KEY, [])
        return [
            DatasetItem.from_json(dataset_item_json)
            for dataset_item_json in dataset_item_jsons
        ]

    @property
    def items(self):
        """Fetch all items belonging to this slice, the type of items returned depends on the type of the slice.
        The type of the slice can be one of { dataset_item, object, scene }.


        Returns: List of DatasetItems for a `dataset_item` slice,
            list of Annotations/Predictions for an `object` slice,
            or a list of Scenes for a `scene` slice.
        """
        try:
            response = self._client.make_request(
                {}, f"slice/{self.id}", requests_command=requests.get
            )
        except NucleusAPIError as e:
            if e.status_code == 503:
                e.message += "/n Your request timed out while trying to get all the items in the slice. Please try slice.items_generator() instead."
            raise e

        items = response.get(ITEMS_KEY, [])

        formatted_items = []
        for item in items:
            item_id_prefix = item["id"].split("_")[0]
            if item_id_prefix == "di":
                formatted_items.append(DatasetItem.from_json(item))
            elif item_id_prefix == "ann":
                formatted_items.append(Annotation.from_json(item))
            elif item_id_prefix == "pred":
                formatted_items.append(prediction_from_json(item))
            elif item_id_prefix == "scn":
                # here we skip validate since no frames for the scene is fetched
                formatted_items.append(
                    Scene.from_json(
                        item, client=self._client, skip_validate=True
                    )
                )
            else:
                raise ValueError("Unknown prefix", item_id_prefix)

        return formatted_items

    def info(self) -> dict:
        """Retrieves the name, slice_id, and dataset_id of the Slice.

        Returns:
            A dict mapping keys to the corresponding info retrieved.
            ::

                {
                    "name": Union[str, int],
                    "slice_id": str,
                    "dataset_id": str,
                    "type": str
                    "pending_job_count": int
                    "created_at": datetime
                    "description": Union[str, None]
                    "tags":
                }
        """
        info = KeyErrorDict(
            items="The 'items' key is now deprecated for Slice.info. Use Slice.items instead."
        )
        res = self._client.make_request(
            {}, f"slice/{self.id}/info", requests_command=requests.get
        )
        info.update(res)
        self._name = info["name"]
        self._dataset_id = info["dataset_id"]
        self._created_at = info["created_at"]
        self._pending_job_count = info["pending_job_count"]
        self._type = info["type"]
        self._description = info["description"]
        return info

    def add_tags(self, tags: List[str]) -> dict:
        """Tag a slice with custom tag names.

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            slc = client.get_slice("YOUR_SLICE_ID")

            slc.add_tags(["tag_1", "tag_2"])

        Args:
            tags: list of tag names
        """
        response = self._client.make_request(
            payload={SLICE_TAGS_KEY: tags},
            route=f"slice/{self.id}/tag",
            requests_command=requests.post,
        )
        return response

    def append(
        self,
        reference_ids: Optional[List[str]] = None,
    ) -> dict:
        """Appends existing DatasetItems from a Dataset to a Slice.

        The endpoint expects a list of DatasetItem reference IDs which are set
        at upload time. The length of reference_ids cannot exceed 10,000 items per request.

        Parameters:
            reference_ids: List of user-defined reference IDs of dataset items or scenes
              to append to the slice.

        Returns:
            Dict of the slice_id and the newly appended IDs. ::

                {
                    "slice_id": str,
                    "new_items": List[str]
                }

        Raises:
            BadRequest: If length of reference_ids is too large (> 10,000 items)
        """
        response = self._client.append_to_slice(
            slice_id=self.id,
            reference_ids=reference_ids,
            dataset_id=self.dataset_id,
        )
        return response

    def items_and_annotation_generator(
        self,
    ) -> Iterable[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Provides a generator of all DatasetItems and Annotations in the slice.

        Returns:
            Generator where each element is a dict containing the DatasetItem
            and all of its associated Annotations, grouped by type (e.g. box).
            ::

                Iterable[{
                    "item": DatasetItem,
                    "annotations": {
                        "box": List[BoxAnnotation],
                        "polygon": List[PolygonAnnotation],
                        "cuboid": List[CuboidAnnotation],
                        "line": List[LineAnnotation],
                        "segmentation": List[SegmentationAnnotation],
                        "category": List[CategoryAnnotation],
                        "keypoints": List[KeypointsAnnotation],
                    }
                }]
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"slice/{self.id}/exportForTrainingPage",
            result_key=EXPORT_FOR_TRAINING_KEY,
            page_size=10000,  # max ES page size
        )
        for data in json_generator:
            for ia in convert_export_payload([data], has_predictions=False):
                yield ia

    def items_and_annotations(
        self,
    ) -> List[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Provides a list of all DatasetItems and Annotations in the Slice.

        Returns:
            List where each element is a dict containing the DatasetItem
            and all of its associated Annotations, grouped by type (e.g. box).
            ::

                List[{
                    "item": DatasetItem,
                    "annotations": {
                        "box": List[BoxAnnotation],
                        "polygon": List[PolygonAnnotation],
                        "cuboid": List[CuboidAnnotation],
                        "line": List[LineAnnotation],
                        "segmentation": List[SegmentationAnnotation],
                        "category": List[CategoryAnnotation],
                        "keypoints": List[KeypointsAnnotation],
                    }
                }]
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"slice/{self.id}/exportForTraining",
            requests_command=requests.get,
        )
        return convert_export_payload(api_payload[EXPORTED_ROWS])

    def export_predictions(
        self, model
    ) -> List[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Provides a list of all DatasetItems and Predictions in the Slice for the given Model.

        Parameters:
            model (Model): the nucleus model objects representing the model for which to export predictions.

        Returns:
            List where each element is a dict containing the DatasetItem
            and all of its associated Predictions, grouped by type (e.g. box).
            ::

                List[{
                    "item": DatasetItem,
                    "predictions": {
                        "box": List[BoxAnnotation],
                        "polygon": List[PolygonAnnotation],
                        "cuboid": List[CuboidAnnotation],
                        "segmentation": List[SegmentationAnnotation],
                        "category": List[CategoryAnnotation],
                    }
                }]
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"slice/{self.id}/{model.id}/exportForTraining",
            requests_command=requests.get,
        )
        return convert_export_payload(api_payload[EXPORTED_ROWS], True)

    def export_raw_json(
        self,
    ) -> List[Union[DatasetItem, Annotation, Prediction, Scene]]:
        """Exports object slices in a raw JSON format. Note that it currently does not support item-level slices.

        For each object or match in an object slice, this method exports the following information:
        - The item that contains the object.
        - The prediction and/or annotation (both, if the slice is based on IOU matches).
        - If the object is part of a scene, it includes scene-level attributes in the export.

        Returns:
            An iterable where each element is a dictionary containing JSON-formatted data.
            ::

                List[{
                    "item": DatasetItem (as JSON),
                    "annotation": BoxAnnotation/CuboidAnnotation (as JSON)
                    "prediction": BoxPrediction/CuboidPrediction (as JSON)
                    "scene": Scene (as JSON)
                    }
                }]
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"slice/{self.id}/export_raw_json",
            requests_command=requests.get,
        )
        return api_payload

    def export_predictions_generator(
        self, model
    ) -> Iterable[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Provides a list of all DatasetItems and Predictions in the Slice for the given Model.

        Parameters:
            model (Model): the nucleus model objects representing the model for which to export predictions.

        Returns:
            Iterable where each element is a dict containing the DatasetItem
            and all of its associated Predictions, grouped by type (e.g. box).
            ::

                List[{
                    "item": DatasetItem,
                    "predictions": {
                        "box": List[BoxAnnotation],
                        "polygon": List[PolygonAnnotation],
                        "cuboid": List[CuboidAnnotation],
                        "segmentation": List[SegmentationAnnotation],
                        "category": List[CategoryAnnotation],
                    }
                }]
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"slice/{self.id}/{model.id}/exportForTrainingPage",
            result_key=EXPORT_FOR_TRAINING_KEY,
            page_size=100000,
        )
        for data in json_generator:
            for ip in convert_export_payload([data], has_predictions=True):
                yield ip

    def export_scale_task_info(self):
        """Fetches info for all linked Scale tasks of items/scenes in the slice.

        Returns:
            A list of dicts, each with two keys, respectively mapping to items/scenes
            and info on their corresponding Scale tasks within the dataset::

                List[{
                    "item" | "scene": Union[DatasetItem, Scene],
                    "scale_task_info": {
                        "task_id": str,
                        "task_status": str,
                        "task_audit_status": str,
                        "task_audit_review_comment": Optional[str],
                        "project_name": str,
                        "batch": str,
                        "created_at": str,
                        "completed_at": Optional[str]
                    }]
                }]

        """
        response = self._client.make_request(
            payload=None,
            route=f"slice/{self.id}/exportScaleTaskInfo",
            requests_command=requests.get,
        )
        # TODO: implement format function with nice keying
        return format_scale_task_info_response(response)

    def send_to_labeling(self, project_id: str):
        """Send items in the Slice as tasks to a Scale labeling project.

        This endpoint submits the items of the Slice as tasks to a pre-existing
        Scale Annotation project uniquely identified by projectId. Only projects
        of type General Image Annotation are currently supported. Additionally,
        in order for task submission to succeed, the project must have task
        instructions and geometries configured as project-level parameters.  In
        order to create a project or set project parameters, you must use the
        Scale Annotation API, which is documented here: `Scale Annotation API
        Documentation <https://docs.scale.com/reference/project-overview>`_.
        When the newly created annotation tasks are annotated, the annotations
        will be automatically reflected in the Nucleus platform.

        For self-serve projects, user can choose to submit the slice as a
        calibration batch, which is recommended for brand new labeling projects.
        For more information about calibration batches, please reference
        `Overview of Self Serve Workflow
        <https://docs.scale.com/reference/batch-overview>`_. Note: A batch can
        be either a calibration batch or a self label batch, but not both.

        Note: Nucleus only supports bounding box, polygon, and line annotations.
        If the project parameters specify any other geometries (ellipses or
        points), those objects will be annotated, but they will not be reflected
        in Nucleus.

        Parameters:
            project_id: Scale-defined ID of the target annotation project.

        .. todo ::
            Add the below parameters, if needed.

            calibration_batch (Optional[bool]): Relevant to Scale Rapid projects
                only.  An optional boolean signaling whether to send as a
                "calibration batch" for taskers to preliminarily evaluate your
                project instructions and parameters.
            self_label_batch (Optional[bool]): Relevant to Scale Rapid projects
                only.  An optional boolean signaling whether to send as a
                "self-label batch," in which your team can label internally
                through Scale Rapid.
        """
        response = self._client.make_request(
            {}, f"slice/{self.id}/{project_id}/send_to_labeling"
        )
        return AsyncJob.from_json(response, self._client)

    def export_embeddings(
        self,
        asynchronous: bool = True,
    ) -> Union[List[Dict[str, Union[str, List[float]]]], EmbeddingsExportJob]:
        """Fetches a pd.DataFrame-ready list of slice embeddings.

        Parameters:
            asynchronous: Whether or not to process the export asynchronously (and
                return an :class:`EmbeddingsExportJob` object). Default is True.

        Returns:
            If synchronous, a list where each element is a columnar mapping::

                List[{
                    "reference_id": str,
                    "embedding_vector": List[float]
                }]

            Otherwise, returns an :class:`EmbeddingsExportJob` object.
        """
        if asynchronous:
            api_payload = self._client.make_request(
                payload=None,
                route=f"dataset/{self.id}/async_export_embeddings",
                requests_command=requests.post,
            )

            return EmbeddingsExportJob.from_json(api_payload, self._client)

        api_payload = self._client.make_request(
            payload=None,
            route=f"slice/{self.id}/embeddings",
            requests_command=requests.get,
        )
        return api_payload

    def export_raw_items(self) -> List[Dict[str, str]]:
        """Fetches a list of accessible URLs for each item in the Slice.

        Returns:
            List where each element is a dict containing a DatasetItem and its
            accessible (signed) Scale URL.
            ::

                List[{
                    "id": str,
                    "ref_id": str,
                    "metadata": Dict[str, Union[str, int]],
                    "original_url": str,
                    "scale_url": str
                }]
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"slice/{self.id}/exportRawItems",
            requests_command=requests.get,
        )
        return api_payload


def check_annotations_are_in_slice(
    annotations: List[Annotation], slice_to_check: Slice
) -> Tuple[bool, Set[str]]:
    """Checks whether the supplied Annotation objects exist in the supplied Slice.

    This endpoint checks whether each Annotation object's reference ID (of the
    parent DatasetItem) exists in the Slice.

    Args:
        annotations: Annnotations with ids referring to targets.
        slice: The slice to check against.

    Returns:
        A tuple of two elements.

        1. True if all Annotations are in the Slice, False otherwise;
        2. List of reference IDs not in the Slice.
    """
    reference_ids_not_found_in_slice = {
        annotation.reference_id
        for annotation in annotations
        if annotation.reference_id is not None
    }.difference(
        {
            item_metadata["ref_id"]
            for item_metadata in slice_to_check.dataset_items()
        }
    )
    if reference_ids_not_found_in_slice:
        annotations_are_in_slice = False
    else:
        annotations_are_in_slice = True

    return (
        annotations_are_in_slice,
        reference_ids_not_found_in_slice,
    )


def create_slice_builder_payload(
    name: str,
    sample_size: int,
    sample_method: Union[str, "SliceBuilderMethods"],
    filters: Optional["SliceBuilderFilters"],
):
    """
    Format the slice builder payload request from the dataclasses
    Args:
        name: Name for the slice being created
        sample_size: Number of items to sample
        sample_method: Method to use for sample the dataset items
        filters: Optional set of filters to apply when collecting the dataset items

    Returns:
        A request friendly payload
    """

    assert (
        sample_method in SliceBuilderMethods
    ), f"Method ${sample_method} not available. Must be one of: {SliceBuilderMethods.options()}"

    # enum or string
    sampleMethod = (
        sample_method.value
        if isinstance(sample_method, SliceBuilderMethods)
        else sample_method
    )

    filter_payload: Dict[str, Union[str, dict]] = {}
    if filters is not None:
        if filters.slice_id is not None:
            filter_payload["sliceId"] = filters.slice_id
        if filters.autotag is not None:
            filter_payload["autotag"] = {
                "autotagId": filters.autotag.autotag_id,
                "range": filters.autotag.score_range,
            }

    return {
        "name": name,
        "sampleSize": sample_size,
        "sampleMethod": sampleMethod,
        "filters": filter_payload,
    }
