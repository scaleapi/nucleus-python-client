import datetime
import os
from enum import Enum
from multiprocessing import Pool
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import requests

from nucleus.annotation_uploader import AnnotationUploader, PredictionUploader
from nucleus.async_job import AsyncJob, EmbeddingsExportJob
from nucleus.chip_utils import fetch_image, generate_offsets, process_chip
from nucleus.embedding_index import EmbeddingIndex
from nucleus.evaluation_match import EvaluationMatch
from nucleus.prediction import from_json as prediction_from_json
from nucleus.track import Track
from nucleus.url_utils import sanitize_string_args
from nucleus.utils import (
    convert_export_payload,
    format_dataset_item_response,
    format_prediction_response,
    format_scale_task_info_response,
    paginate_generator,
    serialize_and_write_to_presigned_url,
)

from .annotation import Annotation, check_all_mask_paths_remote
from .constants import (
    ANNOTATIONS_KEY,
    AUTOTAG_SCORE_THRESHOLD,
    BACKFILL_JOB_KEY,
    BOX_TYPE,
    DATASET_ID_KEY,
    DATASET_IS_SCENE_KEY,
    DATASET_ITEM_IDS_KEY,
    DATASET_ITEMS_KEY,
    DATASET_PRIVACY_MODE_KEY,
    DEFAULT_ANNOTATION_UPDATE_MODE,
    EMBEDDING_DIMENSION_KEY,
    EMBEDDINGS_URL_KEY,
    EXPORT_FOR_TRAINING_KEY,
    EXPORTED_ROWS,
    FRAME_RATE_KEY,
    ITEM_KEY,
    ITEMS_KEY,
    JOB_REQ_LIMIT,
    KEEP_HISTORY_KEY,
    MAX_ES_PAGE_SIZE,
    MESSAGE_KEY,
    NAME_KEY,
    OBJECT_IDS_KEY,
    PROCESSED_URL_KEY,
    REFERENCE_ID_KEY,
    REFERENCE_IDS_KEY,
    REQUEST_ID_KEY,
    SCENE_IDS_KEY,
    SLICE_ID_KEY,
    TRACK_REFERENCE_IDS_KEY,
    TRACKS_KEY,
    TRAINED_SLICE_ID_KEY,
    UPDATE_KEY,
    VIDEO_URL_KEY,
)
from .data_transfer_object.dataset_info import DatasetInfo
from .data_transfer_object.dataset_size import DatasetSize
from .data_transfer_object.scenes_list import ScenesList, ScenesListEntry
from .dataset_item import (
    DatasetItem,
    check_all_paths_remote,
    check_for_duplicate_reference_ids,
    check_items_have_dimensions,
)
from .dataset_item_uploader import DatasetItemUploader
from .deprecation_warning import deprecated
from .errors import NotFoundError, NucleusAPIError
from .job import CustomerJobTypes, jobs_status_overview
from .metadata_manager import ExportMetadataType, MetadataManager
from .payload_constructor import (
    construct_append_scenes_payload,
    construct_model_run_creation_payload,
    construct_taxonomy_payload,
)
from .prediction import Prediction
from .scene import LidarScene, Scene, VideoScene, check_all_scene_paths_remote
from .slice import (
    Slice,
    SliceBuilderFilters,
    SliceBuilderMethods,
    SliceType,
    create_slice_builder_payload,
)
from .upload_response import UploadResponse

if TYPE_CHECKING:
    from . import NucleusClient

# TODO: refactor to reduce this file to under 1000 lines.
# pylint: disable=C0302


WARN_FOR_LARGE_UPLOAD = 10000
WARN_FOR_LARGE_SCENES_UPLOAD = 5


class ObjectQueryType(str, Enum):
    IOU = "iou"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    PREDICTIONS_ONLY = "predictions_only"
    GROUND_TRUTH_ONLY = "ground_truth_only"


class Dataset:
    """Datasets are collections of your data that can be associated with models.

    You can append :class:`DatasetItems<DatasetItem>` or :class:`Scenes<LidarScene>`
    with metadata to your dataset, annotate it with ground truth, and upload
    model predictions to evaluate and compare model performance on your data.

    Make sure that the dataset is set up correctly supporting the required datatype (see code sample below).

    Datasets cannot be instantiated directly and instead must be created via API
    endpoint using :meth:`NucleusClient.create_dataset`, or in the dashboard.

    ::

        import nucleus

        client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)

        # Create new dataset supporting DatasetItems
        dataset = client.create_dataset(YOUR_DATASET_NAME, is_scene=False)

        # OR create new dataset supporting LidarScenes
        dataset = client.create_dataset(YOUR_DATASET_NAME, is_scene=True)

        # Or, retrieve existing dataset by ID
        # This ID can be fetched using client.list_datasets() or from a dashboard URL
        existing_dataset = client.get_dataset("YOUR_DATASET_ID")
    """

    def __init__(
        self,
        dataset_id,
        client: "NucleusClient",
        name=None,
        is_scene=None,
        use_privacy_mode=None,
    ):
        self.id = dataset_id
        self._client = client
        # NOTE: Optionally set name on creation such that the property access doesn't need to hit the server
        self._name = name
        self._is_scene = is_scene
        self._use_privacy_mode = use_privacy_mode

    def __repr__(self):
        if os.environ.get("NUCLEUS_DEBUG", None):
            return f"Dataset(name='{self.name}, dataset_id='{self.id}', is_scene='{self.is_scene}', client={self._client})"
        else:
            return f"Dataset(name='{self.name}, dataset_id='{self.id}', is_scene='{self.is_scene}')"

    def __eq__(self, other):
        if self.id == other.id:
            if self._client == other._client:
                return True
        return False

    @property
    def name(self) -> str:
        """User-defined name of the Dataset."""
        if self._name is None:
            self._name = self._client.make_request(
                {}, f"dataset/{self.id}/name", requests.get
            )["name"]
        return self._name

    @property
    def is_scene(self) -> bool:
        """Whether or not the dataset contains scenes exclusively."""
        if self._is_scene is not None:
            return self._is_scene
        response = self._client.make_request(
            {}, f"dataset/{self.id}/is_scene", requests.get
        )[DATASET_IS_SCENE_KEY]
        self._is_scene = response
        return self._is_scene  # type: ignore

    @property
    def use_privacy_mode(self) -> bool:
        """Whether or not the dataset was created for privacy mode."""
        if self._use_privacy_mode is not None:
            return self._use_privacy_mode
        response = self._client.make_request(
            {}, f"dataset/{self.id}/use_privacy_mode", requests.get
        )[DATASET_PRIVACY_MODE_KEY]
        self._use_privacy_mode = response
        return self._use_privacy_mode  # type: ignore

    @property
    def model_runs(self) -> List[str]:
        """List of all model runs associated with the Dataset."""
        # TODO: model_runs -> models
        response = self._client.make_request(
            {}, f"dataset/{self.id}/model_runs", requests.get
        )
        return response  # type: ignore

    @property
    def slices(self) -> List[Slice]:
        """List of all Slice IDs created from the Dataset."""
        response = self._client.make_request(
            {}, f"dataset/{self.id}/slices", requests.get
        )
        return [Slice.from_request(info, self._client) for info in response]

    @property
    def embedding_indexes(self) -> List[EmbeddingIndex]:
        """Gets all the embedding indexes belonging to this Dataset."""
        response = self._client.make_request(
            {}, f"dataset/{self.id}/embeddingIndexes", requests.get
        )

        return [
            EmbeddingIndex.from_json(info)
            for info in response["embedding_indexes"]
        ]

    def get_slices(
        self,
        name: Optional[str] = None,
        slice_type: Optional[Union[str, SliceType]] = None,
    ) -> List[Slice]:
        """Get a list of slices from its name or underlying slice type.

        Parameters:
            name: Name of the desired slice to look up.
            slice_type: Type of slice to look up. This can be one of ('dataset_item', 'object', 'scene')

        Raises:
            NotFound if no slice(s) were found with the given criteria

        Returns:
            :class:`Slice`: The Nucleus slice as an object.
        """
        endpoint = f"dataset/{self.id}/slices?"

        if name is not None:
            endpoint = f"{endpoint}name={name}&"

        if slice_type is not None:
            if isinstance(slice_type, str):
                slice_type = SliceType(slice_type)

            assert (
                slice_type in SliceType
            ), f"Slice type ${slice_type} is not valid. Must be one of: {SliceType.options()}"
            endpoint = f"{endpoint}type={slice_type}"

        response = self._client.make_request({}, endpoint, requests.get)
        if len(response) == 0:
            errName = f" name={name}" if name is not None else ""
            errType = f"type={slice_type}" if slice_type is not None else ""
            raise NotFoundError(
                f"Slice(s) not found for the parameters:{errName} {errType}"
            )

        return [Slice.from_request(info, self._client) for info in response]

    @property
    def size(self) -> int:
        """Number of items in the Dataset."""
        response = self._client.make_request(
            {}, f"dataset/{self.id}/size", requests.get
        )
        dataset_size = DatasetSize.parse_obj(response)
        return dataset_size.count

    def items_generator(self, page_size=100000) -> Iterable[DatasetItem]:
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
            :class:`DatasetItem`: A single DatasetItem object.
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"dataset/{self.id}/itemsPage",
            result_key=ITEMS_KEY,
            page_size=page_size,
        )
        for item_json in json_generator:
            yield DatasetItem.from_json(item_json)

    @property
    def items(self) -> List[DatasetItem]:
        """List of all DatasetItem objects in the Dataset.

        We recommend using :meth:`Dataset.items_generator` if the Dataset has more than 200k items.
        """
        try:
            response = self._client.make_request(
                {}, f"dataset/{self.id}/datasetItems", requests.get
            )
        except NucleusAPIError as e:
            if e.status_code == 503:
                e.message += "\nThe server timed out while trying to load your items. Please try iterating over dataset.items_generator() instead."
            raise e
        dataset_item_jsons = response.get(DATASET_ITEMS_KEY, None)

        return [
            DatasetItem.from_json(item_json)
            for item_json in dataset_item_jsons
        ]

    @property
    def scenes(self) -> List[ScenesListEntry]:
        """List of ID, reference ID, type, and metadata for all scenes in the Dataset."""
        response = self._client.make_request(
            {}, f"dataset/{self.id}/scenes_list", requests.get
        )

        scenes_list = ScenesList.parse_obj(response)
        return scenes_list.scenes

    @sanitize_string_args
    def autotag_items(self, autotag_name, for_scores_greater_than=0):
        """Fetches the autotag's items above the score threshold, sorted by descending score.

        Parameters:
            autotag_name: The user-defined name of the autotag.
            for_scores_greater_than (Optional[int]): Score threshold between -1
                and 1 above which to include autotag items.

        Returns:
            List of autotagged items above the given score threshold, sorted by
            descending score, and autotag info, packaged into a dict as follows::

                {
                    "autotagItems": List[{
                        ref_id: str,
                        score: float,
                        model_prediction_annotation_id: str | None
                        ground_truth_annotation_id: str | None,
                    }],
                    "autotag": {
                        id: str,
                        name: str,
                        status: "started" | "completed",
                        autotag_level: "Image" | "Object"
                    }
                }

            Note ``model_prediction_annotation_id`` and ``ground_truth_annotation_id``
            are only relevant for object autotags.
        """
        response = self._client.make_request(
            payload={AUTOTAG_SCORE_THRESHOLD: for_scores_greater_than},
            route=f"dataset/{self.id}/autotag/{autotag_name}/taggedItems",
            requests_command=requests.post,
        )
        return response

    def autotag_training_items(self, autotag_name):
        """Fetches items that were manually selected during refinement of the autotag.

        Parameters:
            autotag_name: The user-defined name of the autotag.

        Returns:
            List of user-selected positives and autotag info, packaged into a
            dict as follows::

                {
                    "autotagPositiveTrainingItems": List[{
                        ref_id: str,
                        model_prediction_annotation_id: str | None,
                        ground_truth_annotation_id: str | None,
                    }],
                    "autotag": {
                        id: str,
                        name: str,
                        status: "started" | "completed",
                        autotag_level: "Image" | "Object"
                    }
                }

            Note ``model_prediction_annotation_id`` and ``ground_truth_annotation_id``
            are only relevant for object autotags.
        """
        response = self._client.make_request(
            payload={},
            route=f"dataset/{self.id}/autotag/{autotag_name}/trainingItems",
            requests_command=requests.get,
        )
        return response

    def info(self) -> DatasetInfo:
        """Fetches information about the dataset.

        Returns:
            :class:`DatasetInfo`: Information about the dataset including its
            Scale-generated ID, name, length, associated Models, Slices, and more.
        """
        response = self._client.make_request(
            {}, f"dataset/{self.id}/info", requests.get
        )
        dataset_info = DatasetInfo.parse_obj(response)
        return dataset_info

    @deprecated(
        "Model runs have been deprecated and will be removed. Use a Model instead"
    )
    def create_model_run(
        self,
        name: str,
        reference_id: Optional[str] = None,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        annotation_metadata_schema: Optional[Dict] = None,
    ):
        payload = construct_model_run_creation_payload(
            name,
            reference_id,
            model_id,
            metadata,
            annotation_metadata_schema,
        )
        return self._client.create_model_run(self.id, payload)

    def annotate(
        self,
        annotations: Sequence[Annotation],
        update: bool = DEFAULT_ANNOTATION_UPDATE_MODE,
        batch_size: int = 5000,
        asynchronous: bool = False,
        remote_files_per_upload_request: int = 20,
        local_files_per_upload_request: int = 10,
    ) -> Union[Dict[str, Any], AsyncJob]:
        """Uploads ground truth annotations to the dataset.

        Adding ground truth to your dataset in Nucleus allows you to visualize
        annotations, query dataset items based on the annotations they contain,
        and evaluate models by comparing their predictions to ground truth.

        Nucleus supports :class:`Box<BoxAnnotation>`, :class:`Polygon<PolygonAnnotation>`,
        :class:`Cuboid<CuboidAnnotation>`, :class:`Segmentation<SegmentationAnnotation>`,
        :class:`Category<CategoryAnnotation>`, and :class:`Category<SceneCategoryAnnotation>` annotations.
        Cuboid annotations can only be uploaded to a :class:`pointcloud DatasetItem<LidarScene>`.

        When uploading an annotation, you need to specify which item you are
        annotating via the reference_id you provided when uploading the image
        or pointcloud.

        Ground truth uploads can be made idempotent by specifying an optional
        annotation_id for each annotation. This id should be unique within the
        dataset_item so that (reference_id, annotation_id) is unique within the
        dataset.

        See :class:`SegmentationAnnotation` for specific requirements to upload
        segmentation annotations.

        For ingesting large annotation payloads, see the `Guide for Large Ingestions
        <https://nucleus.scale.com/docs/large-ingestion>`_.

        Parameters:
            annotations (Sequence[:class:`Annotation`]): List of annotation
              objects to upload.
            update: Whether to ignore or overwrite metadata for conflicting annotations.
            batch_size: Number of annotations processed in each concurrent batch.
              Default is 5000. If you get timeouts when uploading geometric annotations,
              you can try lowering this batch size.
            asynchronous: Whether or not to process the upload asynchronously (and
              return an :class:`AsyncJob` object). Default is False.
            remote_files_per_upload_request: Number of remote files to upload in each
                request. Segmentations have either local or remote files, if you are
                getting timeouts while uploading segmentations with remote urls, you
                should lower this value from its default of 20.
            local_files_per_upload_request: Number of local files to upload in each
                request. Segmentations have either local or remote files, if you are
                getting timeouts while uploading segmentations with local files, you
                should lower this value from its default of 10. The maximum is 10.

        Returns:
            If synchronous, payload describing the upload result::

                {
                    "dataset_id": str,
                    "annotations_processed": int
                }

            Otherwise, returns an :class:`AsyncJob` object.
        """
        uploader = AnnotationUploader(dataset_id=self.id, client=self._client)
        uploader.check_for_duplicate_ids(annotations)

        if asynchronous:
            check_all_mask_paths_remote(annotations)
            request_id = serialize_and_write_to_presigned_url(
                annotations, self.id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"dataset/{self.id}/annotate?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        return uploader.upload(
            annotations=annotations,
            update=update,
            batch_size=batch_size,
            remote_files_per_upload_request=remote_files_per_upload_request,
            local_files_per_upload_request=local_files_per_upload_request,
        )

    def ingest_tasks(self, task_ids: List[str]) -> dict:
        """Ingest specific tasks from an existing Scale or Rapid project into the dataset.

        Note: if you would like to create a new Dataset from an exisiting Scale
        labeling project, use :meth:`NucleusClient.create_dataset_from_project`.

        For more info, see our `Ingest From Labeling Guide
        <https://nucleus.scale.com/docs/ingest-from-labeling>`_.

        Parameters:
            task_ids: List of task IDs to ingest.

        Returns:
            Payload describing the asynchronous upload result::

                {
                    "ingested_tasks": int,
                    "ignored_tasks": int,
                    "pending_tasks": int
                }
        """
        # TODO(gunnar): Validate right behaviour. Pydantic?
        return self._client.make_request(
            {"tasks": task_ids}, f"dataset/{self.id}/ingest_tasks"
        )

    def append(
        self,
        items: Union[
            Sequence[DatasetItem], Sequence[LidarScene], Sequence[VideoScene]
        ],
        update: bool = False,
        batch_size: int = 20,
        asynchronous: bool = False,
        local_files_per_upload_request: int = 10,
    ) -> Union[Dict[Any, Any], AsyncJob, UploadResponse]:
        """Appends items or scenes to a dataset.

        .. note::
            Datasets can only accept one of DatasetItems or Scenes, never both.

            This behavior is set during Dataset :meth:`creation
            <NucleusClient.create_dataset>` with the ``is_scene`` flag.

        ::

            import nucleus

            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            dataset = client.get_dataset("YOUR_DATASET_ID")

            local_item = nucleus.DatasetItem(
              image_location="./1.jpg",
              reference_id="image_1",
              metadata={"key": "value"}
            )
            remote_item = nucleus.DatasetItem(
              image_location="s3://your-bucket/2.jpg",
              reference_id="image_2",
              metadata={"key": "value"}
            )

            # default is synchronous upload
            sync_response = dataset.append(items=[local_item])

            # async jobs have higher throughput but can be more difficult to debug
            async_job = dataset.append(
              items=[remote_item], # all items must be remote for async
              asynchronous=True
            )
            print(async_job.status())

        A :class:`Dataset` can be populated with labeled and unlabeled
        data. Using Nucleus, you can filter down the data inside your dataset
        using custom metadata about your images.

        For instance, your local dataset may contain ``Sunny``, ``Foggy``, and
        ``Rainy`` folders of images. All of these images can be uploaded into a
        single Nucleus ``Dataset``, with (queryable) metadata like ``{"weather":
        "Sunny"}``.

        To update an item's metadata, you can re-ingest the same items with the
        ``update`` argument set to true. Existing metadata will be overwritten
        for ``DatasetItems`` in the payload that share a ``reference_id`` with a
        previously uploaded ``DatasetItem``. To retrieve your existing
        ``reference_ids``, use :meth:`Dataset.items`.

        ::

            # overwrite metadata by reuploading the item
            remote_item.metadata["weather"] = "Sunny"

            async_job_2 = dataset.append(
              items=[remote_item],
              update=True,
              asynchronous=True
            )

        Parameters:
            items: ( \
              Union[ \
                Sequence[:class:`DatasetItem`], \
                Sequence[:class:`LidarScene`] \
                Sequence[:class:`VideoScene`]
              ]): List of items or scenes to upload.
            batch_size: Size of the batch for larger uploads. Default is 20. This is
                for items that have a remote URL and do not require a local upload.
                If you get timeouts for uploading remote urls, try decreasing this.
            update: Whether or not to overwrite metadata on reference ID collision.
              Default is False.
            asynchronous: Whether or not to process the upload asynchronously (and
              return an :class:`AsyncJob` object). This is required when uploading
              scenes. Default is False.
            files_per_upload_request: Optional; default is 10. We recommend lowering
                this if you encounter timeouts.
            local_files_per_upload_request: Optional; default is 10. We recommend
                lowering this if you encounter timeouts.
        Returns:
            For scenes
                If synchronous, returns a payload describing the upload result::

                    {
                        "dataset_id: str,
                        "new_items": int,
                        "updated_items": int,
                        "ignored_items": int,
                        "upload_errors": int
                    }

                Otherwise, returns an :class:`AsyncJob` object.
            For images
                If synchronous returns :class:`nucleus.upload_response.UploadResponse`
                otherwise :class:`AsyncJob`
        """
        assert (
            batch_size is None or batch_size < 30
        ), "Please specify a batch size smaller than 30 to avoid timeouts."
        dataset_items = [
            item for item in items if isinstance(item, DatasetItem)
        ]
        lidar_scenes = [item for item in items if isinstance(item, LidarScene)]
        video_scenes = [item for item in items if isinstance(item, VideoScene)]

        check_for_duplicate_reference_ids(dataset_items)

        if self.use_privacy_mode:
            check_items_have_dimensions(dataset_items)

        if dataset_items and (lidar_scenes or video_scenes):
            raise Exception(
                "You must append either DatasetItems or Scenes to the dataset."
            )
        if lidar_scenes:
            assert (
                asynchronous
            ), "In order to avoid timeouts, you must set asynchronous=True when uploading 3D scenes."

            return self._append_scenes(lidar_scenes, update, asynchronous)
        if video_scenes:
            assert (
                asynchronous
            ), "In order to avoid timeouts, you must set asynchronous=True when uploading videos."

            return self._append_video_scenes(
                video_scenes, update, asynchronous
            )

        if len(dataset_items) > WARN_FOR_LARGE_UPLOAD and not asynchronous:
            print(
                "Tip: for large uploads, get faster performance by importing your data "
                "into Nucleus directly from a cloud storage provider. See "
                "https://dashboard.scale.com/nucleus/docs/api?language=python#guide-for-large-ingestions"
                " for details."
            )

        if asynchronous:
            check_all_paths_remote(dataset_items)
            request_id = serialize_and_write_to_presigned_url(
                dataset_items, self.id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"dataset/{self.id}/append?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        return self._upload_items(
            dataset_items,
            update=update,
            batch_size=batch_size,
            local_files_per_upload_request=local_files_per_upload_request,
        )

    @deprecated("Prefer using Dataset.append instead.")
    def append_scenes(
        self,
        scenes: List[LidarScene],
        update: Optional[bool] = False,
        asynchronous: Optional[bool] = False,
    ) -> Union[dict, AsyncJob]:
        return self._append_scenes(scenes, update, asynchronous)

    def _append_scenes(
        self,
        scenes: List[LidarScene],
        update: Optional[bool] = False,
        asynchronous: Optional[bool] = False,
    ) -> Union[dict, AsyncJob]:
        # TODO: make private in favor of Dataset.append invocation
        if not self.is_scene:
            raise Exception(
                "Your dataset is not a scene dataset but only supports single dataset items. "
                "In order to be able to add scenes, please create another dataset with "
                "client.create_dataset(<dataset_name>, is_scene=True) or add the scenes to "
                "an existing scene dataset."
            )

        for scene in scenes:
            scene.validate()

        if not asynchronous:
            print(
                "WARNING: Processing lidar pointclouds usually takes several seconds. As a result, sychronous scene upload"
                "requests are likely to timeout. For large uploads, we recommend using the flag asynchronous=True "
                "to avoid HTTP timeouts. Please see"
                "https://dashboard.scale.com/nucleus/docs/api?language=python#guide-for-large-ingestions"
                " for details."
            )

        if asynchronous:
            check_all_scene_paths_remote(scenes)
            request_id = serialize_and_write_to_presigned_url(
                scenes, self.id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"{self.id}/upload_scenes?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        payload = construct_append_scenes_payload(scenes, update)
        response = self._client.make_request(
            payload=payload,
            route=f"{self.id}/upload_scenes",
        )
        return response

    def _append_video_scenes(
        self,
        scenes: List[VideoScene],
        update: Optional[bool] = False,
        asynchronous: Optional[bool] = False,
    ) -> Union[dict, AsyncJob]:
        if not self.is_scene:
            raise Exception(
                "Your dataset is not a scene dataset but only supports single dataset items. "
                "In order to be able to add scenes, please create another dataset with "
                "client.create_dataset(<dataset_name>, is_scene=True) or add the scenes to "
                "an existing scene dataset."
            )

        for scene in scenes:
            scene.use_privacy_mode = self.use_privacy_mode
            scene.validate()

        if not asynchronous:
            print(
                "WARNING: Processing videos usually takes several seconds. As a result, synchronous video scene upload"
                "requests are likely to timeout. For large uploads, we recommend using the flag asynchronous=True "
                "to avoid HTTP timeouts. Please see"
                "https://dashboard.scale.com/nucleus/docs/api?language=python#guide-for-large-ingestions"
                " for details."
            )

        if asynchronous:
            check_all_scene_paths_remote(scenes)
            request_id = serialize_and_write_to_presigned_url(
                scenes, self.id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"{self.id}/upload_video_scenes?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        payload = construct_append_scenes_payload(scenes, update)
        response = self._client.make_request(
            payload=payload,
            route=f"{self.id}/upload_video_scenes",
        )
        return response

    def iloc(self, i: int) -> dict:
        """Fetches dataset item and associated annotations by absolute numerical index.

        Parameters:
            i: Absolute numerical index of the dataset item within the dataset.

        Returns:
            Payload describing the dataset item and associated annotations::

                {
                    "item": DatasetItem
                    "annotations": {
                        "box": Optional[List[BoxAnnotation]],
                        "cuboid": Optional[List[CuboidAnnotation]],
                        "line": Optional[List[LineAnnotation]],
                        "polygon": Optional[List[PolygonAnnotation]],
                        "keypoints": Optional[List[KeypointsAnnotation]],
                        "segmentation": Optional[List[SegmentationAnnotation]],
                        "category": Optional[List[CategoryAnnotation]],
                    }
                }
        """
        response = self._client.make_request(
            {}, f"dataset/{self.id}/iloc/{i}", requests.get
        )
        return format_dataset_item_response(response)

    @sanitize_string_args
    def refloc(self, reference_id: str) -> dict:
        """Fetches a dataset item and associated annotations by reference ID.

        Parameters:
            reference_id: User-defined reference ID of the dataset item.

        Returns:
            Payload containing the dataset item and associated annotations::

                {
                    "item": DatasetItem
                    "annotations": {
                        "box": Optional[List[BoxAnnotation]],
                        "cuboid": Optional[List[CuboidAnnotation]],
                        "line": Optional[List[LineAnnotation]],
                        "polygon": Optional[List[PolygonAnnotation]],
                        "keypoints": Option[List[KeypointsAnnotation]],
                        "segmentation": Optional[List[SegmentationAnnotation]],
                        "category": Optional[List[CategoryAnnotation]],
                    }
                }
        """
        response = self._client.make_request(
            {}, f"dataset/{self.id}/refloc/{reference_id}", requests.get
        )
        return format_dataset_item_response(response)

    def loc(self, dataset_item_id: str) -> dict:
        """Fetches a dataset item and associated annotations by Nucleus-generated ID.

        Parameters:
            dataset_item_id: Nucleus-generated dataset item ID (starts with ``di_``).
              This can be retrieved via :meth:`Dataset.items` or a Nucleus dashboard URL.

        Returns:
            Payload containing the dataset item and associated annotations::

                {
                    "item": DatasetItem
                    "annotations": {
                        "box": Optional[List[BoxAnnotation]],
                        "cuboid": Optional[List[CuboidAnnotation]],
                        "line": Optional[List[LineAnnotation]],
                        "polygon": Optional[List[PolygonAnnotation]],
                        "keypoints": Optional[List[KeypointsAnnotation]],
                        "segmentation": Optional[List[SegmentationAnnotation]],
                        "category": Optional[List[CategoryAnnotation]],
                    }
                }
        """
        response = self._client.make_request(
            {}, f"dataset/{self.id}/loc/{dataset_item_id}", requests.get
        )
        return format_dataset_item_response(response)

    def ground_truth_loc(self, reference_id: str, annotation_id: str):
        """Fetches a single ground truth annotation by ID.

        Parameters:
            reference_id: User-defined reference ID of the dataset item associated
              with the ground truth annotation.
            annotation_id: User-defined ID of the ground truth annotation.

        Returns:
            Union[\
                :class:`BoxAnnotation`, \
                :class:`LineAnnotation`, \
                :class:`PolygonAnnotation`, \
                :class:`KeypointsAnnotation`, \
                :class:`CuboidAnnotation`, \
                :class:`SegmentationAnnotation` \
                :class:`CategoryAnnotation` \
            ]: Ground truth annotation object with the specified annotation ID.
        """
        response = self._client.make_request(
            {},
            f"dataset/{self.id}/groundTruth/loc/{reference_id}/{annotation_id}",
            requests.get,
        )
        return Annotation.from_json(response)

    def create_slice(
        self,
        name: str,
        reference_ids: List[str],
    ) -> Slice:
        """Creates a :class:`Slice` of dataset items within a dataset.

        Parameters:
            name: A human-readable name for the slice.
            reference_ids: List of reference IDs of dataset items to add to the slice, cannot exceed 10,000 items.

        Returns:
            :class:`Slice`: The newly constructed slice item.

        Raises:
            BadRequest: If length of reference_ids is too large (> 10,000 items)
        """
        payload = {NAME_KEY: name, REFERENCE_IDS_KEY: reference_ids}
        response = self._client.make_request(
            payload, f"dataset/{self.id}/create_slice"
        )
        return Slice(response[SLICE_ID_KEY], self._client)

    def create_slice_by_ids(
        self,
        name: str,
        dataset_item_ids: Optional[List[str]] = None,
        scene_ids: Optional[List[str]] = None,
        annotation_ids: Optional[List[str]] = None,
        prediction_ids: Optional[List[str]] = None,
    ) -> Slice:
        """Creates a :class:`Slice` of dataset items, scenes, annotations, or predictions within a dataset by their IDs.

        .. note::
            Dataset item, scene, and object (annotation or prediction) IDs may not be mixed.
            However, when creating an object slice, both annotation and prediction IDs may be supplied.

        Parameters:
            name: A human-readable name for the slice.
            dataset_item_ids: List of internal IDs of dataset items to add to the slice::
            scene_ids: List of internal IDs of scenes to add to the slice::
            annotation_ids: List of internal IDs of Annotations to add to the slice::
            prediction_ids: List of internal IDs of Predictions to add to the slice::

        Returns:
            :class:`Slice`: The newly constructed slice item.
        """
        if (
            not dataset_item_ids
            and not scene_ids
            and not annotation_ids
            and not prediction_ids
        ):
            raise Exception("Must provide at least one list of internal IDs")
        payload = {
            NAME_KEY: name,
            DATASET_ITEM_IDS_KEY: dataset_item_ids
            if dataset_item_ids
            else None,
            SCENE_IDS_KEY: scene_ids if scene_ids else None,
            OBJECT_IDS_KEY: [*(annotation_ids or []), *(prediction_ids or [])]
            or None,
        }
        response = self._client.make_request(
            payload, f"dataset/{self.id}/create_slice"
        )
        return Slice(response[SLICE_ID_KEY], self._client)

    def build_slice(
        self,
        name: str,
        sample_size: int,
        sample_method: Union[str, SliceBuilderMethods],
        filters: Optional[SliceBuilderFilters] = None,
    ) -> Union[str, Tuple[AsyncJob, str], dict]:
        """Build a slice using Nucleus' Smart Sample tool. Allowing slices to be built
        based on certain criteria, and filters.

        Args:
            name: Name for the slice being created. Must be unique per dataset.
            sample_size: Size of the slice to create. Capped by the size of the dataset and the applied filters.
            sample_method: How to sample the dataset, currently supports 'Random' and 'Uniqueness'
            filters: Apply filters to only sample from an existing slice or autotag

        Examples:
            from nucleus.slice import SliceBuilderFilters, SliceBuilderMethods, SliceBuilderFilterAutotag

            # random slice
            job = dataset.build_slice("RandomSlice", 20, SliceBuilderMethods.RANDOM)

            # slice with filters
            filters = SliceBuilderFilters(
                slice_id="<some slice id>",
                autotag=SliceBuilderFilterAutotag("tag_cd41jhjdqyti07h8m1n1", [-0.5, 0.5])
            )
            job = dataset.build_slice("NewSlice", 20, SliceBuilderMethods.RANDOM, filters)

        Returns: An async job

        """
        payload = create_slice_builder_payload(
            name, sample_size, sample_method, filters
        )

        response = self._client.make_request(
            payload,
            f"dataset/{self.id}/build_slice",
        )

        slice_id = ""
        if "sliceId" in response:
            slice_id = response["sliceId"]
        if "job_id" in response:
            return AsyncJob.from_json(response, self._client), slice_id
        return response

    @sanitize_string_args
    def delete_item(self, reference_id: str) -> dict:
        """Deletes an item from the dataset by item reference ID.

        All annotations and predictions associated with the item will be deleted
        as well.

        Parameters:
            reference_id: The user-defined reference ID of the item to delete.

        Returns:
            Payload to indicate deletion invocation.
        """
        return self._client.make_request(
            {},
            f"dataset/{self.id}/refloc/{reference_id}",
            requests.delete,
        )

    @sanitize_string_args
    def delete_scene(self, reference_id: str):
        """Deletes a sene from the Dataset by scene reference ID.

        All items, annotations, and predictions associated with the scene will be
        deleted as well.

        Parameters:
            reference_id: The user-defined reference ID of the item to delete.
        """
        self._client.delete(f"dataset/{self.id}/scene/{reference_id}")

    def list_autotags(self):
        # TODO: prefer Dataset.autotags @property
        """Fetches all autotags of the dataset.

        Returns:
            List of autotag payloads::

                List[{
                    "id": str,
                    "name": str,
                    "status": "completed" | "pending",
                    "autotag_level": "Image" | "Object"
                }]
        """
        return self._client.list_autotags(self.id)

    def update_autotag(self, autotag_id: str) -> AsyncJob:
        """Rerun autotag inference on all items in the dataset.

        Currently this endpoint does not try to skip already inferenced items,
        but this improvement is planned for the future. This means that for
        now, you can only have one job running at a time, so please await the
        result using job.sleep_until_complete() before launching another job.

        Parameters:
            autotag_id: ID of the autotag to re-inference. You can retrieve the
                ID you want with :meth:`list_autotags`, or from its URL in the
                "Manage Autotags" page in the dashboard.

        Returns:
          :class:`AsyncJob`: Asynchronous job object to track processing status.
        """
        return AsyncJob.from_json(
            payload=self._client.make_request(
                {}, f"autotag/{autotag_id}", requests.post
            ),
            client=self._client,
        )

    def create_custom_index(
        self, embeddings_urls: List[str], embedding_dim: int
    ):
        """Processes user-provided embeddings for the dataset to use with autotag and simsearch.

        ::

            import nucleus

            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            dataset = client.get_dataset("YOUR_DATASET_ID")

            all_embeddings = {
                "reference_id_0": [0.1, 0.2, 0.3],
                "reference_id_1": [0.4, 0.5, 0.6],
                ...
                "reference_id_10000": [0.7, 0.8, 0.9]
            } # sharded and uploaded to s3 with the two below URLs

            embeddings_url_1 = "s3://dataset/embeddings_map_1.json"
            embeddings_url_2 = "s3://dataset/embeddings_map_2.json"

            response = dataset.create_custom_index(
                embeddings_url=[embeddings_url_1, embeddings_url_2],
                embedding_dim=3
            )

        Parameters:
            embeddings_urls:  List of URLs, each of which pointing to
              a JSON mapping reference_id -> embedding vector. Each embedding JSON must
              contain <5000 rows.
            embedding_dim: The dimension of the embedding vectors. Must be consistent
              across all embedding vectors in the index.

        Returns:
            :class:`AsyncJob`: Asynchronous job object to track processing status.
        """
        res = self._client.post(
            {
                EMBEDDINGS_URL_KEY: embeddings_urls,
                EMBEDDING_DIMENSION_KEY: embedding_dim,
            },
            f"indexing/{self.id}",
        )
        return AsyncJob.from_json(
            res,
            self._client,
        )

    def delete_custom_index(self, image: bool = True):
        """Deletes the custom index uploaded to the dataset.

        Returns:
            Payload containing information that can be used to track the job's status::

                {
                    "dataset_id": str,
                    "job_id": str,
                    "message": str
                }
        """
        return self._client.delete_custom_index(self.id, image)

    def set_primary_index(self, image: bool = True, custom: bool = False):
        """Sets the primary index used for Autotag and Similarity Search on this dataset.

        Parameters:
            image: Whether to configure the primary index for images or objects.
                Default is True (set primary image index).
            custom: Whether to set the primary index to use custom or Nucleus-generated
                embeddings. Default is True (use custom embeddings as the primary index).
        Returns:

            {
                "success": bool,
            }
        """
        return self._client.set_primary_index(self.id, image, custom)

    def set_continuous_indexing(self, enable: bool = True):
        """Toggle whether embeddings are automatically generated for new data.

        Sets continuous indexing for a given dataset, which will automatically
        generate embeddings for use with autotag whenever new images are uploaded.

        Parameters:
            enable: Whether to enable or disable continuous indexing. Default is
              True.

        Returns:
            Response payload::

                {
                    "dataset_id": str,
                    "message": str
                    "backfill_job": AsyncJob,
                }
        """
        preprocessed_response = self._client.set_continuous_indexing(
            self.id, enable
        )
        response = {
            DATASET_ID_KEY: preprocessed_response[DATASET_ID_KEY],
            MESSAGE_KEY: preprocessed_response[MESSAGE_KEY],
        }
        if enable:
            response[BACKFILL_JOB_KEY] = AsyncJob.from_json(
                preprocessed_response, self._client
            )

        return response

    def get_image_indexing_status(self):
        """Gets the primary image index progress for the dataset.

        Returns:
            Response payload::

                {
                    "embedding_count": int
                    "image_count": int
                    "percent_indexed": float
                    "additional_context": str
                }
        """
        return self._client.make_request(
            {"image": True},
            f"dataset/{self.id}/indexingStatus",
            requests_command=requests.post,
        )

    def get_object_indexing_status(self, model_run_id=None):
        """Gets the primary object index progress of the dataset.
        If model_run_id is not specified, this endpoint will retrieve the indexing progress of the ground truth objects.

        Returns:
            Response payload::

                {
                    "embedding_count": int
                    "object_count": int
                    "percent_indexed": float
                    "additional_context": str
                }
        """
        return self._client.make_request(
            {"image": False, "model_run_id": model_run_id},
            f"dataset/{self.id}/indexingStatus",
            requests_command=requests.post,
        )

    def create_image_index(self):
        """Creates or updates image index by generating embeddings for images that do not already have embeddings.

        The embeddings are used for autotag and similarity search.

        This endpoint is limited to index up to 2 million images at a time and the
        job will fail for payloads that exceed this limit.

        Returns:
            :class:`AsyncJob`: Asynchronous job object to track processing status.
        """
        response = self._client.create_image_index(self.id)
        return AsyncJob.from_json(response, self._client)

    def create_object_index(
        self,
        model_run_id: Optional[str] = None,
        gt_only: Optional[bool] = None,
    ):
        """Creates or updates object index by generating embeddings for objects that do not already have embeddings.

        These embeddings are used for autotag and similarity search. This endpoint
        only supports indexing objects sourced from the predictions of a specific
        model or the ground truth annotations of the dataset.

        This endpoint is idempotent. If this endpoint is called again for a model
        whose predictions were indexed in the past, the previously indexed predictions
        will not have new embeddings recomputed. The same is true for ground truth
        annotations.

        Note that this means if you change update a prediction or ground truth
        bounding box that already has an associated embedding, the embedding will
        not be updated, even with another call to this endpoint. For now, we
        recommend deleting the prediction or ground truth annotation and
        re-inserting it to force generate a new embedding.

        This endpoint is limited to generating embeddings for 3 million objects
        at a time and the job will fail for payloads that exceed this limit.

        Parameters:
            model_run_id: The ID of the model whose predictions should be indexed.
              Default is None, but must be supplied in the absence of ``gt_only``.

              .. todo ::
                  Deprecate model run

            gt_only: Whether to only generate embeddings for the ground truth
              annotations of the dataset. Default is None, but must be supplied
              in the absence of ``model_run_id``.

        Returns:
            :class:`AsyncJob`: Asynchronous job object to track processing status.
        """
        response = self._client.create_object_index(
            self.id, model_run_id, gt_only
        )
        return AsyncJob.from_json(response, self._client)

    def add_taxonomy(
        self,
        taxonomy_name: str,
        taxonomy_type: str,
        labels: List[str],
        update: bool = False,
    ):
        """Creates a new taxonomy.

        At the moment we only support taxonomies for category annotations and
        predictions.

        ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            dataset = client.get_dataset("ds_bwkezj6g5c4g05gqp1eg")

            response = dataset.add_taxonomy(
                taxonomy_name="clothing_type",
                taxonomy_type="category",
                labels=["shirt", "trousers", "dress"],
                update=False
            )

        Parameters:
            taxonomy_name: The name of the taxonomy. Taxonomy names must be
              unique within a dataset.
            taxonomy_type: The type of this taxonomy as a string literal.
              Currently, the only supported taxonomy type is "category."
            labels: The list of possible labels for the taxonomy.
            update: Whether or not to update taxonomy labels on taxonomy name
              collision. Default is False. Note that taxonomy labels will not be
              deleted on update, they can only be appended.

        Returns:
            Returns a response with dataset_id, taxonomy_name, and status of the
            add taxonomy operation.

            ::

                {
                    "dataset_id": str,
                    "taxonomy_name": str,
                    "status": "Taxonomy created"
                }
        """
        return self._client.make_request(
            construct_taxonomy_payload(
                taxonomy_name, taxonomy_type, labels, update
            ),
            f"dataset/{self.id}/add_taxonomy",
            requests_command=requests.post,
        )

    def delete_taxonomy(
        self,
        taxonomy_name: str,
    ):
        """Deletes the given taxonomy.

        All annotations and predictions associated with the taxonomy will be
        deleted as well.

        Parameters:
            taxonomy_name: The name of the taxonomy.

        Returns:
            Returns a response with dataset_id, taxonomy_name, and status of the
            delete taxonomy operation.

            ::

                {
                    "dataset_id": str,
                    "taxonomy_name": str,
                    "status": "Taxonomy successfully deleted"
                }
        """
        return self._client.make_request(
            {},
            f"dataset/{self.id}/taxonomy/{taxonomy_name}",
            requests.delete,
        )

    def items_and_annotations(
        self,
    ) -> List[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Returns a list of all DatasetItems and Annotations in this dataset.

        Returns:
            A list of dicts, each with two keys representing a row in the dataset::

                List[{
                    "item": DatasetItem,
                    "annotations": {
                        "box": Optional[List[BoxAnnotation]],
                        "cuboid": Optional[List[CuboidAnnotation]],
                        "line": Optional[List[LineAnnotation]],
                        "polygon": Optional[List[PolygonAnnotation]],
                        "segmentation": Optional[List[SegmentationAnnotation]],
                        "category": Optional[List[CategoryAnnotation]],
                        "keypoints": Optional[List[KeypointsAnnotation]],
                    }
                }]
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"dataset/{self.id}/exportForTraining",
            requests_command=requests.get,
        )
        return convert_export_payload(api_payload[EXPORTED_ROWS])

    def items_and_annotation_generator(
        self,
        query: Optional[str] = None,
    ) -> Iterable[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Provides a generator of all DatasetItems and Annotations in the dataset.

        Args:
            query: Structured query compatible with the `Nucleus query language <https://nucleus.scale.com/docs/query-language-reference>`_.

        Returns:
            Generator where each element is a dict containing the DatasetItem
            and all of its associated Annotations, grouped by type.
            ::

                Iterable[{
                    "item": DatasetItem,
                    "annotations": {
                        "box": List[BoxAnnotation],
                        "polygon": List[PolygonAnnotation],
                        "cuboid": List[CuboidAnnotation],
                        "line": Optional[List[LineAnnotation]],
                        "segmentation": List[SegmentationAnnotation],
                        "category": List[CategoryAnnotation],
                        "keypoints": List[KeypointsAnnotation],
                    }
                }]
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"dataset/{self.id}/exportForTrainingPage",
            result_key=EXPORT_FOR_TRAINING_KEY,
            page_size=10000,  # max ES page size
            query=query,
        )
        for data in json_generator:
            for ia in convert_export_payload([data], has_predictions=False):
                yield ia

    def items_and_annotation_chip_generator(
        self,
        chip_size: int,
        stride_size: int,
        cache_directory: str,
        query: Optional[str] = None,
        num_processes: int = 0,
    ) -> Iterable[Dict[str, str]]:
        """Provides a generator of chips for all DatasetItems and BoxAnnotations in the dataset.

        A chip is an image created by tiling a source image.

        Args:
            chip_size: The size of the image chip
            stride_size: The distance to move when creating the next image chip.
              When stride is equal to chip size, there will be no overlap.
              When stride is equal to half the chip size, there will be 50 percent overlap.
            cache_directory: The s3 or local directory to store the image and annotations of a chip.
              s3 directories must be in the format s3://s3-bucket/s3-key
            query: Structured query compatible with the `Nucleus query language <https://nucleus.scale.com/docs/query-language-reference>`_.
            num_processes: The number of worker processes to use to chip and upload images. If unset, no parallel processing will occur.

        Returns:
            Generator where each element is a dict containing the location of the image chip (jpeg) and its annotations (json).
            ::

                Iterable[{
                    "image_location": str,
                    "annotation_location": str
                }]
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"dataset/{self.id}/exportForTrainingPage",
            result_key=EXPORT_FOR_TRAINING_KEY,
            page_size=10000,  # max ES page size
            query=query,
            chip=True,
        )
        for item in json_generator:
            image = fetch_image(item[ITEM_KEY][PROCESSED_URL_KEY])
            w, h = image.size
            annotations = item[BOX_TYPE]
            item_ref_id = item[ITEM_KEY][REFERENCE_ID_KEY]
            offsets = generate_offsets(w, h, chip_size, stride_size)
            chip_args = [
                (
                    offset,
                    chip_size,
                    w,
                    h,
                    item_ref_id,
                    cache_directory,
                    image,
                    annotations,
                )
                for offset in offsets
            ]
            if num_processes:
                with Pool(num_processes) as pool:
                    for chip_result in pool.imap(process_chip, chip_args):
                        yield chip_result
            else:
                for chip_arg in chip_args:
                    yield process_chip(chip_arg)

    def export_embeddings(
        self,
        asynchronous: bool = True,
    ) -> Union[List[Dict[str, Union[str, List[float]]]], EmbeddingsExportJob]:
        """Fetches a pd.DataFrame-ready list of dataset embeddings.

        Parameters:
            asynchronous: Whether or not to process the export asynchronously (and
                return an :class:`EmbeddingsExportJob` object). Default is True.

        Returns:
            If synchronous, a list where each item is a dict with two keys representing a row
            in the dataset::

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
            route=f"dataset/{self.id}/embeddings",
            requests_command=requests.get,
        )
        return api_payload  # type: ignore

    def delete_annotations(
        self, reference_ids: Optional[list] = None, keep_history: bool = True
    ) -> AsyncJob:
        """Deletes all annotations associated with the specified item reference IDs.

        Parameters:
            reference_ids: List of user-defined reference IDs of the dataset items
              from which to delete annotations. Defaults to an empty list.
            keep_history: Whether to preserve version history. We recommend
              skipping this parameter and using the default value of True.
        Returns:
            :class:`AsyncJob`: Empty payload response.
        """
        if reference_ids is None:
            reference_ids = []
        payload = {
            KEEP_HISTORY_KEY: keep_history,
            REFERENCE_IDS_KEY: reference_ids,
        }
        response = self._client.make_request(
            payload,
            f"annotation/{self.id}",
            requests_command=requests.delete,
        )
        return AsyncJob.from_json(response, self._client)

    def get_scene(self, reference_id: str) -> Scene:
        """Fetches a single scene in the dataset by its reference ID.

        Parameters:
            reference_id: The user-defined reference ID of the scene to fetch.

        Returns:
            :class:`Scene<LidarScene>`: A scene object containing frames, which
            in turn contain pointcloud or image items.
        """
        response = self._client.make_request(
            payload=None,
            route=f"dataset/{self.id}/scene/{reference_id}",
            requests_command=requests.get,
        )
        if FRAME_RATE_KEY in response or VIDEO_URL_KEY in response:
            return VideoScene.from_json(response, self._client)
        return LidarScene.from_json(response, self._client)

    def get_scene_from_item_ref_id(
        self, item_reference_id: str
    ) -> Optional[Scene]:
        """Given a dataset item reference ID, find the Scene it belongs to."""
        if not self.is_scene:
            print(
                f"Dataset {self.id} is not a scene. Cannot call this endpoint."
            )
            return None

        response = self._client.make_request(
            payload=None,
            route=f"dataset/{self.id}/scene/{item_reference_id}?from_item=1",
            requests_command=requests.get,
        )
        if FRAME_RATE_KEY in response or VIDEO_URL_KEY in response:
            return VideoScene.from_json(response, self._client)
        return LidarScene.from_json(response, self._client)

    def export_predictions(self, model):
        """Fetches all predictions of a model that were uploaded to the dataset.

        Parameters:
            model (:class:`Model`): The model whose predictions to retrieve.

        Returns:
            List[Union[\
                :class:`BoxPrediction`, \
                :class:`PolygonPrediction`, \
                :class:`CuboidPrediction`, \
                :class:`SegmentationPrediction` \
                :class:`CategoryPrediction`, \
                :class:`KeypointsPrediction`, \
            ]]: List of prediction objects from the model.

        """
        json_response = self._client.make_request(
            payload=None,
            route=f"dataset/{self.id}/model/{model.id}/export",
            requests_command=requests.get,
        )
        return format_prediction_response({ANNOTATIONS_KEY: json_response})

    def export_scale_task_info(self):
        """Fetches info for all linked Scale tasks of items/scenes in the dataset.

        Returns:
            A list of dicts, each with two keys, respectively mapping to items/scenes
            and info on their corresponding Scale tasks within the dataset::

                List[{
                    "item" | "scene": Union[:class:`DatasetItem`, :class:`Scene`],
                    "scale_task_info": {
                        "task_id": str,
                        "task_status": str,
                        "task_audit_status": str,
                        "task_audit_review_comment": Optional[str],
                        "project_name": str,
                        "batch": str,
                        "created_at": str,
                        "completed_at": Optional[str]
                    }[]
                }]

        """
        response = self._client.make_request(
            payload=None,
            route=f"dataset/{self.id}/exportScaleTaskInfo",
            requests_command=requests.get,
        )
        return format_scale_task_info_response(response)

    def calculate_evaluation_metrics(
        self, model, options: Optional[dict] = None
    ):
        """Starts computation of evaluation metrics for a model on the dataset.

        To update matches and metrics calculated for a model on a given dataset you
        can call this endpoint. This is required in order to sort by IOU, view false
        positives/false negatives, and view model insights.

        You can add predictions from a model to a dataset after running the
        calculation of the metrics. However, the calculation of metrics will have
        to be retriggered for the new predictions to be matched with ground truth
        and appear as false positives/negatives, or for the new predictions effect
        on metrics to be reflected in model run insights.

        During IoU calculation, bounding box Predictions are compared to
        GroundTruth using a greedy matching algorithm that matches prediction and
        ground truth boxes that have the highest ious first. By default the
        matching algorithm is class-agnostic: it will greedily create matches
        regardless of the class labels.

        The algorithm can be tuned to classify true positives between certain
        classes, but not others. This is useful if the labels in your ground truth
        do not match the exact strings of your model predictions, or if you want
        to associate multiple predictions with one ground truth label, or multiple
        ground truth labels with one prediction. To recompute metrics based on
        different matching, you can re-commit the run with new request parameters.

        ::

            import nucleus

            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            dataset = client.get_dataset(dataset_id="YOUR_DATASET_ID")

            model = client.get_model(model_id="YOUR_MODEL_PRJ_ID")

            # Compute all evaluation metrics including IOU-based matching:
            dataset.calculate_evaluation_metrics(model)

            # Match car and bus bounding boxes (for IOU computation)
            # Otherwise enforce that class labels must match
            dataset.calculate_evaluation_metrics(model, options={
              'allowed_label_matches': [
                {
                  'ground_truth_label': 'car',
                  'model_prediction_label': 'bus'
                },
                {
                  'ground_truth_label': 'bus',
                  'model_prediction_label': 'car'
                }
              ]
            })

        Parameters:
            model (:class:`Model`): The model object for which to calculate metrics.
            options: Dictionary of specific options to configure metrics calculation.

                class_agnostic
                  Whether ground truth and prediction classes can differ when
                  being matched for evaluation metrics. Default is True.

                allowed_label_matches
                  Pairs of ground truth and prediction classes that should
                  be considered matchable when computing metrics. If supplied,
                  ``class_agnostic`` must be False.

                ::

                    {
                        "class_agnostic": bool,
                        "allowed_label_matches": List[{
                            "ground_truth_label": str,
                            "model_prediction_label": str
                        }]
                    }
        """
        if options is None:
            options = {}
        return self._client.make_request(
            payload=options,
            route=f"dataset/{self.id}/model/{model.id}/calculateEvaluationMetrics",
        )

    def upload_predictions(
        self,
        model,
        predictions: List[Prediction],
        update: bool = False,
        asynchronous: bool = False,
        batch_size: int = 5000,
        remote_files_per_upload_request: int = 20,
        local_files_per_upload_request: int = 10,
        trained_slice_id: Optional[str] = None,
    ):
        """Uploads predictions and associates them with an existing :class:`Model`.

        Adding predictions to your dataset in Nucleus allows you to visualize
        discrepancies against ground truth, query dataset items based on the
        predictions they contain, and evaluate your models by comparing their
        predictions to ground truth.

        Nucleus supports :class:`Box<BoxPrediction>`, :class:`Polygon<PolygonPrediction>`,
        :class:`Cuboid<CuboidPrediction>`, :class:`Segmentation<SegmentationPrediction>`,
        :class:`Category<CategoryPrediction>`, and :class:`Category<SceneCategoryPrediction>` predictions. Cuboid predictions
        can only be uploaded to a :class:`pointcloud DatasetItem<LidarScene>`.

        When uploading a prediction, you need to specify which item you are
        annotating via the reference_id you provided when uploading the image
        or pointcloud.

        Ground truth uploads can be made idempotent by specifying an optional
        annotation_id for each prediction. This id should be unique within the
        dataset_item so that (reference_id, annotation_id) is unique within the
        dataset.

        See :class:`SegmentationPrediction` for specific requirements to upload
        segmentation predictions.

        For ingesting large prediction payloads, see the `Guide for Large Ingestions
        <https://nucleus.scale.com/docs/large-ingestion>`_.

        Parameters:
            model (:class:`Model`): Nucleus-generated model ID (starts with ``prj_``). This can
              be retrieved via :meth:`list_models` or a Nucleus dashboard URL.
            predictions (List[Union[\
                :class:`BoxPrediction`, \
                :class:`PolygonPrediction`, \
                :class:`CuboidPrediction`, \
                :class:`SegmentationPrediction`, \
                :class:`CategoryPrediction` \
                :class:`SceneCategoryPrediction` \
            ]]): List of prediction objects to upload.
            update: Whether or not to overwrite metadata or ignore on reference ID
              collision. Default is False.
            asynchronous: Whether or not to process the upload asynchronously (and
              return an :class:`AsyncJob` object). Default is False.
            batch_size: Number of predictions processed in each concurrent batch.
              Default is 5000. If you get timeouts when uploading geometric predictions,
              you can try lowering this batch size. This is only relevant for
              asynchronous=False
            remote_files_per_upload_request: Number of remote files to upload in each
              request. Segmentations have either local or remote files, if you are
              getting timeouts while uploading segmentations with remote urls, you
              should lower this value from its default of 20. This is only relevant for
              asynchronous=False.
            local_files_per_upload_request: Number of local files to upload in each
              request. Segmentations have either local or remote files, if you are
              getting timeouts while uploading segmentations with local files, you
              should lower this value from its default of 10. The maximum is 10.
              This is only relevant for asynchronous=False
            trained_slice_id: Nucleus-generated slice ID (starts with ``slc_``) which was used
                to train the model.

        Returns:
            Payload describing the synchronous upload::
                {
                    "dataset_id": str,
                    "model_run_id": str,
                    "predictions_processed": int,
                    "predictions_ignored": int,
                }
        """
        uploader = PredictionUploader(
            model_run_id=None,
            dataset_id=self.id,
            model_id=model.id,
            client=self._client,
        )
        uploader.check_for_duplicate_ids(predictions)

        if asynchronous:
            check_all_mask_paths_remote(predictions)

            request_id = serialize_and_write_to_presigned_url(
                predictions, self.id, self._client
            )
            response = self._client.make_request(
                payload={
                    REQUEST_ID_KEY: request_id,
                    UPDATE_KEY: update,
                    TRAINED_SLICE_ID_KEY: trained_slice_id,
                },
                route=f"dataset/{self.id}/model/{model.id}/uploadPredictions?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        return uploader.upload(
            annotations=predictions,
            batch_size=batch_size,
            update=update,
            remote_files_per_upload_request=remote_files_per_upload_request,
            local_files_per_upload_request=local_files_per_upload_request,
            trained_slice_id=trained_slice_id,
        )

    def predictions_iloc(self, model, index):
        """Fetches all predictions of a dataset item by its absolute index.

        Parameters:
            model (:class:`Model`): Model object from which to fetch the prediction.
            index (int): Absolute index of the dataset item within the dataset.

        Returns:
            List[Union[\
                :class:`BoxPrediction`, \
                :class:`PolygonPrediction`, \
                :class:`CuboidPrediction`, \
                :class:`SegmentationPrediction` \
                :class:`CategoryPrediction`, \
                :class:`KeypointsPrediction`, \
            ]]: Dictionary mapping prediction type to a list of such prediction
            objects from the given model::

                {
                    "box": List[BoxPrediction],
                    "polygon": List[PolygonPrediction],
                    "cuboid": List[CuboidPrediction],
                    "segmentation": List[SegmentationPrediction],
                    "category": List[CategoryPrediction],
                    "keypoints": List[KeypointsPrediction],
                }
        """
        return format_prediction_response(
            self._client.make_request(
                payload=None,
                route=f"dataset/{self.id}/model/{model.id}/iloc/{index}",
                requests_command=requests.get,
            )
        )

    def predictions_refloc(self, model, reference_id):
        """Fetches all predictions of a dataset item by its reference ID.

        Parameters:
            model (:class:`Model`): Model object from which to fetch the prediction.
            reference_id (str): User-defined ID of the dataset item from which to fetch
              all predictions.

        Returns:
            List[Union[\
                :class:`BoxPrediction`, \
                :class:`PolygonPrediction`, \
                :class:`CuboidPrediction`, \
                :class:`SegmentationPrediction` \
                :class:`CategoryPrediction`, \
                :class:`KeypointsPrediction`, \
            ]]: Dictionary mapping prediction type to a list of such prediction
            objects from the given model::

                {
                    "box": List[BoxPrediction],
                    "polygon": List[PolygonPrediction],
                    "cuboid": List[CuboidPrediction],
                    "segmentation": List[SegmentationPrediction],
                    "category": List[CategoryPrediction],
                    "keypoints": List[KeypointsPrediction],
                }
        """
        return format_prediction_response(
            self._client.make_request(
                payload=None,
                route=f"dataset/{self.id}/model/{model.id}/referenceId/{reference_id}",
                requests_command=requests.get,
            )
        )

    def prediction_loc(self, model, reference_id, annotation_id):
        """Fetches a single ground truth annotation by id.

        Parameters:
            model (:class:`Model`): Model object from which to fetch the prediction.
            reference_id (str): User-defined reference ID of the dataset item
              associated with the model prediction.
            annotation_id (str): User-defined ID of the ground truth annotation.

        Returns:
            Union[\
                :class:`BoxPrediction`, \
                :class:`PolygonPrediction`, \
                :class:`CuboidPrediction`, \
                :class:`SegmentationPrediction` \
                :class:`CategoryPrediction` \
                :class:`KeypointsPrediction` \
            ]: Model prediction object with the specified annotation ID.
        """
        return prediction_from_json(
            self._client.make_request(
                payload=None,
                route=f"dataset/{self.id}/model/{model.id}/loc/{reference_id}/{annotation_id}",
                requests_command=requests.get,
            )
        )

    def _upload_items(
        self,
        dataset_items: List[DatasetItem],
        batch_size: int = 20,
        update: bool = False,
        local_files_per_upload_request: int = 10,
    ) -> UploadResponse:
        """
        Appends images to a dataset with given dataset_id.
        Overwrites images on collision if updated.

        Args:
            dataset_items: Items to Upload
            batch_size: how many items with remote urls to include in each request.
                If you get timeouts for uploading remote urls, try decreasing this.
            update: Update records on conflict otherwise overwrite
            local_files_per_upload_request: How large to make each upload request when your
                files are local. If you get timeouts, you may need to lower this from
                its default of 10. The maximum is 10.

        Returns:
            UploadResponse
        """
        if self.is_scene:
            raise Exception(
                "Your dataset is a scene dataset and does not support the upload of single dataset items. "
                "In order to be able to add dataset items, please create another dataset with "
                "client.create_dataset(<dataset_name>, is_scene=False) or add the dataset items to "
                "an existing dataset supporting dataset items."
            )
        uploader = DatasetItemUploader(self.id, self._client)
        return uploader.upload(
            dataset_items=dataset_items,
            batch_size=batch_size,
            update=update,
            local_files_per_upload_request=local_files_per_upload_request,
        )

    def update_scene_metadata(
        self, mapping: Dict[str, dict], asynchronous: bool = False
    ):
        """
        Update (merge) scene metadata for each reference_id given in the mapping.
        The backend will join the specified mapping metadata to the existing metadata.
        If there is a key-collision, the value given in the mapping will take precedence.

        Args:
            mapping: key-value pair of <reference_id>: <metadata>
            asynchronous: if True, run the update as a background job

        Examples:
            >>> mapping = {"scene_ref_1": {"new_key": "foo"}, "scene_ref_2": {"some_value": 123}}
            >>> dataset.update_scene_metadata(mapping)

        Returns:
            A dictionary outlining success or failures.
        """
        mm = MetadataManager(
            self.id,
            self._client,
            mapping,
            ExportMetadataType.SCENES,
            asynchronous,
        )
        return mm.update()

    def update_item_metadata(
        self, mapping: Dict[str, dict], asynchronous: bool = False
    ):
        """
        Update (merge) dataset item metadata for each reference_id given in the mapping.
        The backend will join the specified mapping metadata to the existing metadata.
        If there is a key-collision, the value given in the mapping will take precedence.

        This method may also be used to udpate the `camera_params` for a particular set of items.
        Just specify the key `camera_params` in the metadata for each reference_id along with all the necessary fields.

        Args:
            mapping: key-value pair of <reference_id>: <metadata>
            asynchronous: if True, run the update as a background job

        Examples:
            >>> mapping = {"item_ref_1": {"new_key": "foo"}, "item_ref_2": {"some_value": 123, "camera_params": {...}}}
            >>> dataset.update_item_metadata(mapping)

        Returns:
            A dictionary outlining success or failures.
        """
        mm = MetadataManager(
            self.id,
            self._client,
            mapping,
            ExportMetadataType.DATASET_ITEMS,
            asynchronous,
        )
        return mm.update()

    def query_items(self, query: str) -> Iterable[DatasetItem]:
        """
        Fetches all DatasetItems that pertain to a given structured query.

        Args:
            query: Structured query compatible with the `Nucleus query language <https://nucleus.scale.com/docs/query-language-reference>`_.

        Returns:
            A list of DatasetItem query results.
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"dataset/{self.id}/queryItemsPage",
            result_key=ITEMS_KEY,
            page_size=MAX_ES_PAGE_SIZE,
            query=query,
        )
        for item_json in json_generator:
            yield DatasetItem.from_json(item_json)

    def query_scenes(self, query: str) -> Iterable[Scene]:
        """
        Fetches all Scenes that pertain to a given structured query.

        Args:
            query: Structured query compatible with the `Nucleus query language <https://nucleus.scale.com/docs/query-language-reference>`_.

        Returns:
            A list of Scene query results.
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"dataset/{self.id}/queryScenesPage",
            result_key=ITEMS_KEY,
            page_size=MAX_ES_PAGE_SIZE,
            query=query,
        )
        for item_json in json_generator:
            yield Scene.from_json(item_json, None, True)

    def query_objects(
        self,
        query: str,
        query_type: ObjectQueryType,
        model_run_id: Optional[str] = None,
    ) -> Iterable[Union[Annotation, Prediction, EvaluationMatch]]:
        """
        Fetches all objects in the dataset that pertain to a given structured query.
        The results are either Predictions, Annotations, or Evaluation Matches, based on the objectType input parameter

        Args:
            query: Structured query compatible with the `Nucleus query language <https://nucleus.scale.com/docs/query-language-reference>`_.
            objectType: Defines the type of the object to query

        Returns:
            An iterable of either Predictions, Annotations, or Evaluation Matches
        """
        json_generator = paginate_generator(
            client=self._client,
            endpoint=f"dataset/{self.id}/queryObjectsPage",
            result_key=ITEMS_KEY,
            page_size=MAX_ES_PAGE_SIZE,
            query=query,
            patch_mode=query_type,
            model_run_id=model_run_id,
        )

        for item_json in json_generator:
            if query_type == ObjectQueryType.GROUND_TRUTH_ONLY:
                yield Annotation.from_json(item_json)
            elif query_type == ObjectQueryType.PREDICTIONS_ONLY:
                yield prediction_from_json(item_json)
            elif query_type in [
                ObjectQueryType.IOU,
                ObjectQueryType.FALSE_POSITIVE,
                ObjectQueryType.FALSE_NEGATIVE,
            ]:
                yield EvaluationMatch.from_json(item_json)
            else:
                raise ValueError("Unknown object type", query_type)

    @property
    def tracks(self) -> List[Track]:
        """Tracks unique to this dataset.

        Returns:
            List of `Track` objects.
        """
        response = self._client.make_request(
            {}, f"dataset/{self.id}/tracks", requests.get
        )

        tracks_list = [
            Track.from_json(
                payload=track,
                connection=self._client.connection,
            )
            for track in response[TRACKS_KEY]
        ]
        return tracks_list

    def delete_tracks(self, track_reference_ids: List[str]) -> None:
        """
        Deletes a list of tracks from the dataset, thereby unlinking their annotation and prediction instances.

        Parameters:
            reference_ids (List[str]): A list of reference IDs for tracks to delete.
        """

        self._client.make_request(
            payload={
                TRACK_REFERENCE_IDS_KEY: track_reference_ids,
            },
            route=f"dataset/{self.id}/tracks",
            requests_command=requests.delete,
        )

    def jobs(
        self,
        job_types: Optional[List[CustomerJobTypes]] = None,
        from_date: Optional[Union[str, datetime.datetime]] = None,
        to_date: Optional[Union[str, datetime.datetime]] = None,
        limit: int = JOB_REQ_LIMIT,
        show_completed: bool = False,
        stats_only: bool = False,
    ):
        """
        Fetch jobs pertaining to this particular dataset.

        Parameters:
            job_types: Filter on set of job types, if None, fetch all types, ie: ['uploadDatasetItems']
            from_date: beginning of date range, as a string 'YYYY-MM-DD' or datetime object.
                For example: '2021-11-05', parser.parse('Nov 5 2021'), or datetime(2021,11,5)
            to_date: end of date range
            limit: number of results to fetch, max 50_000
            show_completed: dont fetch jobs with Completed status
            stats_only: return overview of jobs, instead of a list of job objects
        """
        job_objects = self._client.list_jobs(
            dataset_id=self.id,
            show_completed=show_completed,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            job_types=job_types,
        )
        if stats_only:
            return jobs_status_overview(job_objects)
        return job_objects
