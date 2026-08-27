"""Nucleus Python SDK."""

__all__ = [
    "AsyncJob",
    "AllowedLabelMatch",
    "Benchmark",
    "BenchmarkItemsPage",
    "TrainingSet",
    "TrainingSetItemsPage",
    "EmbeddingsExportJob",
    "BoxAnnotation",
    "DeduplicationJob",
    "DeduplicationResult",
    "DeduplicationStats",
    "LocalDeduplicationResult",
    "BoxPrediction",
    "CameraParams",
    "CategoryAnnotation",
    "CategoryPrediction",
    "CuboidAnnotation",
    "CuboidPrediction",
    "Dataset",
    "DatasetInfo",
    "DatasetItem",
    "DatasetItemRetrievalError",
    "EvaluationV2",
    "EvaluationV2Charts",
    "EvaluationV2ExamplesPage",
    "EvaluationV2FilterArgs",
    "EvaluationV2FilterSchema",
    "EvaluationV2MatchExample",
    "EvaluationV2Preset",
    "EvaluationV2Status",
    "LeaderboardF1CurveEntry",
    "LeaderboardRankingEntry",
    "MetadataExclusionRule",
    "LabelExclusionRule",
    "BoxAreaExclusionRule",
    "RollupGroup",
    "Frame",
    "Keypoint",
    "KeypointsAnnotation",
    "KeypointsPrediction",
    "LidarPoint",
    "LidarScene",
    "LineAnnotation",
    "LinePrediction",
    "Model",
    "ModelCreationError",
    "ModelWeights",
    # "MultiCategoryAnnotation", # coming soon!
    "NotFoundError",
    "NucleusAPIError",
    "NucleusClient",
    "Point",
    "Point3D",
    "PolygonAnnotation",
    "PolygonPrediction",
    "Quaternion",
    "SceneCategoryAnnotation",
    "SceneCategoryPrediction",
    "Segment",
    "SegmentationAnnotation",
    "SegmentationPrediction",
    "Slice",
    "VideoScene",
    "deduplicate_by_phash",
]

import datetime
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import requests
import tqdm

if TYPE_CHECKING:
    # Backwards compatibility is even uglier with mypy
    from pydantic.v1 import parse_obj_as
else:
    try:
        # NOTE: we always use pydantic v1 but have to do these shenanigans to support both v1 and v2
        from pydantic.v1 import parse_obj_as
    except ImportError:
        from pydantic import parse_obj_as

from nucleus.url_utils import sanitize_string_args

from . import metrics
from .annotation import (
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    Keypoint,
    KeypointsAnnotation,
    LidarPoint,
    LineAnnotation,
    MultiCategoryAnnotation,
    Point,
    Point3D,
    PolygonAnnotation,
    SceneCategoryAnnotation,
    Segment,
    SegmentationAnnotation,
)
from .async_job import AsyncJob, EmbeddingsExportJob
from .async_utils import make_multiple_requests_concurrently
from .benchmark import Benchmark
from .camera_params import CameraParams
from .connection import Connection
from .constants import (
    ALLOWED_LABEL_MATCHES_CAMEL_KEY,
    ANNOTATION_METADATA_SCHEMA_KEY,
    ANNOTATIONS_IGNORED_KEY,
    ANNOTATIONS_PROCESSED_KEY,
    AUTOTAGS_KEY,
    BENCHMARK_ID_KEY,
    BENCHMARK_IDS_KEY,
    BUMP_TYPE_KEY,
    COLLAPSE_KEY,
    CONFIDENCE_THRESHOLD_KEY,
    DATASET_ID_KEY,
    DATASET_IDS_KEY,
    DATASET_IS_SCENE_KEY,
    DATASET_PRIVACY_MODE_KEY,
    DEFAULT_NETWORK_TIMEOUT_SEC,
    DELETED_KEY,
    DESCRIPTION_KEY,
    DRAFT_KEY,
    EMBEDDING_DIMENSION_KEY,
    EMBEDDINGS_URL_KEY,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    ERRORS_KEY,
    EVALUATION_ID_KEY,
    EXCLUSION_RULES_CAMEL_KEY,
    GLOB_SIZE_THRESHOLD_CHECK,
    HEIGHT_KEY,
    I_KEY,
    IMAGE_KEY,
    IMAGE_LOCATION_KEY,
    IMAGE_URL_KEY,
    INDEX_CONTINUOUS_ENABLE_KEY,
    ITEM_IDS_KEY,
    ITEM_METADATA_SCHEMA_KEY,
    ITEMS_KEY,
    JOB_CREATION_TIME_KEY,
    JOB_ID_KEY,
    JOB_LAST_KNOWN_STATUS_KEY,
    JOB_TYPE_KEY,
    KEEP_HISTORY_KEY,
    MESSAGE_KEY,
    METADATA_KEY,
    METRIC_TYPE_KEY,
    MODEL_IDS_KEY,
    MODEL_RUN_ID_KEY,
    MODEL_RUN_IDS_KEY,
    MODEL_TAGS_KEY,
    MODEL_TRAINED_SLICE_IDS_KEY,
    NAME_KEY,
    NUCLEUS_ENDPOINT,
    PARENT_BENCHMARK_ID_KEY,
    PARENT_TRAINING_SET_ID_KEY,
    POINTCLOUD_LOCATION_KEY,
    POINTCLOUD_URL_KEY,
    POINTS_KEY,
    PREDICTIONS_IGNORED_KEY,
    PREDICTIONS_PROCESSED_KEY,
    REFERENCE_IDS_KEY,
    REMOVED_ITEM_IDS_KEY,
    ROLLUP_GROUPS_CAMEL_KEY,
    SCENE_IDS_KEY,
    SCOPE_KEY,
    SLICE_ID_KEY,
    SLICE_IDS_KEY,
    SLICE_TAGS_KEY,
    STATUS_CODE_KEY,
    TOP_N_KEY,
    TRAINING_SET_ID_KEY,
    TRAINING_SET_IDS_KEY,
    UPDATE_KEY,
    UPLOAD_ID_KEY,
    URL_KEY,
    VERSION_LABEL_KEY,
    VERSION_MAJOR_KEY,
    VERSION_MINOR_KEY,
    WIDTH_KEY,
)
from .data_transfer_object.dataset_details import DatasetDetails
from .data_transfer_object.dataset_info import DatasetInfo
from .data_transfer_object.evaluation_v2 import (
    BenchmarkItemsPage,
    EvaluationV2Charts,
    EvaluationV2ExamplesPage,
    EvaluationV2FilterArgs,
    EvaluationV2FilterSchema,
    EvaluationV2MatchExample,
    LeaderboardF1CurveEntry,
    LeaderboardRankingEntry,
)
from .data_transfer_object.job_status import JobInfoRequestPayload
from .data_transfer_object.training_set import TrainingSetItemsPage
from .dataset import Dataset
from .dataset_item import DatasetItem
from .deduplication import (
    DeduplicationJob,
    DeduplicationResult,
    DeduplicationStats,
)
from .deprecation_warning import deprecated
from .errors import (
    DatasetItemRetrievalError,
    ModelCreationError,
    ModelRunCreationError,
    NoAPIKey,
    NotFoundError,
    NucleusAPIError,
)
from .evaluation_v2 import (
    AllowedLabelMatch,
    EvaluationV2,
    EvaluationV2Status,
    RollupGroup,
)
from .evaluation_v2_exclusions import (
    BoxAreaExclusionRule,
    EvaluationV2ExclusionRule,
    LabelExclusionRule,
    MetadataExclusionRule,
)
from .evaluation_v2_preset import _UNSET, EvaluationV2Preset
from .job import CustomerJobTypes
from .local_deduplication import (
    LocalDeduplicationResult,
    deduplicate_by_phash,
)
from .model import Model
from .model_run import ModelRun
from .model_weights import (
    MODEL_WEIGHTS_MAX_BYTES,
    ModelWeights,
    _finalize_payload,
    _presign_payload,
    _progress_to_bar,
    _stream_weights_to_file,
    _transfer_weights_to_storage,
)
from .payload_constructor import (
    construct_annotation_payload,
    construct_box_predictions_payload,
    construct_model_creation_payload,
    construct_segmentation_payload,
)
from .prediction import (
    BoxPrediction,
    CategoryPrediction,
    CuboidPrediction,
    KeypointsPrediction,
    LinePrediction,
    PolygonPrediction,
    SceneCategoryPrediction,
    SegmentationPrediction,
)
from .quaternion import Quaternion
from .retry_strategy import RetryStrategy
from .scene import Frame, LidarScene, VideoScene
from .slice import Slice
from .training_set import TrainingSet
from .utils import create_items_from_folder_crawl
from .validate import Validate

# pylint: disable=E1101
# TODO: refactor to reduce this file to under 1000 lines.
# pylint: disable=C0302


class NucleusClient:
    """Client to interact with the Nucleus API via Python SDK.

    Parameters:
        api_key: One of ``api_key`` or ``limited_access_key`` must be provided; you cannot pass
          both. For standard Scale API key authentication, pass the key here. Follow `this guide
          <https://scale.com/docs/api-reference/authentication>`_ to retrieve API keys. If you omit
          this argument and are not using ``limited_access_key``, the SDK falls back to the
          ``NUCLEUS_API_KEY`` environment variable.
        limited_access_key: Nucleus-only API key for scoped access. Reach out to your Scale
          representative to obtain a limited access key.
        use_notebook: Whether the client is being used in a notebook (toggles tqdm
          style). Default is ``False``.
        endpoint: Base URL of the API. Default is Nucleus's current production API.

    .. note::

        You must provide **either** a standard Scale API key (``api_key``, or
        ``NUCLEUS_API_KEY`` in the environment) **or** a Nucleus-only key
        (``limited_access_key``), never both. Passing both arguments, or setting
        the environment variable ``NUCLEUS_API_KEY`` while also passing
        ``limited_access_key``, will raise an error.

    Example::

        # Using a basic auth key
        import nucleus
        client = nucleus.NucleusClient(api_key="YOUR_API_KEY", ...)

        # Using only a limited access key (no Basic Auth)
        import nucleus
        client = nucleus.NucleusClient(limited_access_key="YOUR_LIMITED_KEY", ...)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_notebook: bool = False,
        endpoint: Optional[str] = None,
        limited_access_key: Optional[str] = None,
    ):
        effective_basic_key = (
            api_key if api_key else os.environ.get("NUCLEUS_API_KEY")
        )
        if limited_access_key and effective_basic_key:
            raise ValueError(
                "Cannot provide both 'api_key' and 'limited_access_key'. "
                "Use 'api_key' for standard Scale API key authentication, "
                "or 'limited_access_key' for Nucleus-only access, but not both."
            )
        # Allow usage with only a limited access key
        if api_key is None and limited_access_key:
            self.api_key = None
        else:
            self.api_key = self._set_api_key(api_key)
        self.tqdm_bar = tqdm.tqdm
        if endpoint is None:
            self.endpoint = os.environ.get(
                "NUCLEUS_ENDPOINT", NUCLEUS_ENDPOINT
            )
        else:
            self.endpoint = endpoint
        self._use_notebook = use_notebook
        if use_notebook:
            import tqdm.notebook as tqdm_notebook

            self.tqdm_bar = tqdm_notebook.tqdm
        self.extra_headers: Dict[str, str] = {}
        if limited_access_key:
            self.extra_headers["x-limited-access-key"] = limited_access_key
        self.connection = Connection(
            self.api_key, self.endpoint, extra_headers=self.extra_headers
        )
        self.validate = Validate(
            self.api_key, self.endpoint, extra_headers=self.extra_headers
        )

    def __repr__(self):
        return f"NucleusClient(api_key='{self.api_key}', use_notebook={self._use_notebook}, endpoint='{self.endpoint}')"

    def __eq__(self, other):
        if self.api_key == other.api_key:
            if self._use_notebook == other._use_notebook:
                return True
        return False

    @property
    def datasets(self) -> List[Dataset]:
        """List all Datasets

        Returns:
            List of all datasets accessible to user
        """
        response = self.make_request({}, "dataset/details", requests.get)
        dataset_details = (
            parse_obj_as(  # pylint: disable=used-before-assignment
                List[DatasetDetails], response
            )
        )
        return [
            Dataset(d.id, client=self, name=d.name) for d in dataset_details
        ]

    @property
    def models(self) -> List[Model]:
        # TODO: implement for Dataset, scoped just to associated models
        """Fetches all of your Nucleus models.

        Returns:
            List[:class:`Model`]: List of models associated with the client API key.
        """
        model_objects = self.make_request({}, "models/", requests.get)

        return [
            Model(
                model_id=model["id"],
                name=model["name"],
                reference_id=model["ref_id"],
                metadata=model["metadata"] or None,
                client=self,
                tags=model.get(MODEL_TAGS_KEY, []),
                trained_slice_ids=model.get(MODEL_TRAINED_SLICE_IDS_KEY, None),
            )
            for model in model_objects["models"]
        ]

    @property
    def jobs(
        self,
    ) -> List[AsyncJob]:
        """Lists all jobs, see NucleusClient.list_jobs(...) for advanced options

        Returns:
            List of all AsyncJobs
        """
        return self.list_jobs()

    @property
    def slices(self) -> List[Slice]:
        response = self.make_request({}, "slice/", requests.get)
        slices = [Slice.from_request(info, self) for info in response]
        return slices

    @deprecated(msg="Use the NucleusClient.models property in the future.")
    def list_models(self) -> List[Model]:
        return self.models

    @deprecated(msg="Use the NucleusClient.datasets property in the future.")
    def list_datasets(self) -> Dict[str, Union[str, List[str]]]:
        return self.make_request({}, "dataset/", requests.get)

    def list_jobs(
        self,
        show_completed: bool = False,
        from_date: Optional[Union[str, datetime.datetime]] = None,
        to_date: Optional[Union[str, datetime.datetime]] = None,
        job_types: Optional[List[CustomerJobTypes]] = None,
        limit: Optional[int] = None,
        dataset_id: Optional[str] = None,
        date_limit: Optional[str] = None,
    ) -> List[AsyncJob]:
        """Fetches all of your running jobs in Nucleus.

        Parameters:
            show_completed: Whether to include jobs with Completed status.
            from_date: Beginning of date range filter.
            to_date: End of date range filter.
            job_types: Filter on set of job types. If None, fetch all types.
            limit: Number of results to fetch, max 50,000.
            dataset_id: Filter on a particular dataset.
            date_limit: Deprecated, do not use.

        Returns:
            List[:class:`AsyncJob`]: List of running asynchronous jobs
            associated with the client API key.
        """

        if date_limit is not None:
            warnings.warn(
                "Argument `date_limit` is no longer supported. Consider using the `from_date` and `to_date` args."
            )

        payload = JobInfoRequestPayload(
            dataset_id=dataset_id,
            show_completed=show_completed,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            job_types=job_types,
        ).dict()

        job_objects = self.make_request(payload, "jobs/", requests.post)
        return [
            AsyncJob(
                job_id=job[JOB_ID_KEY],
                job_last_known_status=job[JOB_LAST_KNOWN_STATUS_KEY],
                job_type=job[JOB_TYPE_KEY],
                job_creation_time=job[JOB_CREATION_TIME_KEY],
                client=self,
            )
            for job in job_objects
        ]

    @deprecated(msg="Prefer using Dataset.items")
    def get_dataset_items(self, dataset_id) -> List[DatasetItem]:
        dataset = self.get_dataset(dataset_id)
        return dataset.items

    def get_dataset(self, dataset_id: str) -> Dataset:
        """Fetches a dataset by its ID.

        Parameters:
            dataset_id: The ID of the dataset to fetch.

        Returns:
            :class:`Dataset`: The Nucleus dataset as an object.
        """
        return Dataset(dataset_id, self)

    def get_job(self, job_id: str) -> AsyncJob:
        """Fetches a job by its ID.

        Parameters:
            job_id: The ID of the job to fetch.

        Returns:
            :class:`AsyncJob`: The Nucleus async job as an object.
        """
        payload = self.make_request(
            payload={},
            route=f"job/{job_id}/info",
            requests_command=requests.get,
        )
        return AsyncJob.from_json(payload=payload, client=self)

    def get_model(
        self,
        model_id: Optional[str] = None,
        model_run_id: Optional[str] = None,
    ) -> Model:
        """Fetches a model by its ID.

        Parameters:
            model_id: You can pass either a model ID (starts with ``prj_``) or a
                model run ID (starts with ``run_``). Retrieved via :meth:`list_models`
                or from a Nucleus dashboard URL.
            model_run_id: You can pass either a model ID (starts with ``prj_``), or a
                model run ID (starts with ``run_``). This can be retrieved via
                :meth:`list_models` or a Nucleus dashboard URL. Model run IDs result
                from the application of a model to a dataset. In the future, we plan
                to hide ``model_run_ids`` fully from users.

        Returns:
            :class:`Model`: The Nucleus model as an object.
        """
        if model_id is None and model_run_id is None:
            raise ValueError("Must pass either a model_id or a model_run_id")
        if model_id is not None and model_run_id is not None:
            raise ValueError("Must pass either a model_id or a model_run_id")

        model_or_model_run_id = (
            model_id if model_id is not None else model_run_id
        )

        payload = self.make_request(
            payload={},
            route=f"model/{model_or_model_run_id}",
            requests_command=requests.get,
        )
        return Model.from_json(payload=payload, client=self)

    @deprecated(
        "Model runs have been deprecated and will be removed. Use a Model instead"
    )
    def get_model_run(self, model_run_id: str, dataset_id: str) -> ModelRun:
        return ModelRun(model_run_id, dataset_id, self)

    @deprecated(
        "Model runs have been deprecated and will be removed. Use a Model instead"
    )
    def delete_model_run(self, model_run_id: str):
        return self.make_request(
            {}, f"modelRun/{model_run_id}", requests.delete
        )

    def merge_model_runs(
        self,
        model_run_ids: List[str],
        *,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge several model runs into one new run holding all their predictions.

        A benchmark evaluation names a single model run, and a benchmark's items may
        span several datasets. A model whose predictions were uploaded as separate runs
        — one per dataset, or one per inference batch — therefore has no single run
        covering the benchmark, and every uncovered item scores as a false negative.
        Merging the runs produces one run that does cover it, which you can then pass to
        :meth:`create_benchmark_evaluation_v2`.

        All source runs must belong to the same model.

        The merge is a full union of all predictions. If two
        source runs predict on the same item with the same ``annotation_id``, the
        colliding id is rewritten rather than dropped. The source runs are left
        untouched.

        **Asynchronous.** The new run is created and returned immediately, but its
        predictions are copied by a background job. The run is *empty until the job
        completes*, so wait on the returned job before evaluating::

            result = client.merge_model_runs(["run_abc", "run_def"])
            result["job"].sleep_until_complete()
            client.create_benchmark_evaluation_v2(
                benchmark_id, result["model_run_id"]
            )

        Parameters:
            model_run_ids: Two or more distinct model run ids (``run_*``) to merge.
            name: Display name for the merged run. Defaults server-side to the model's
                own name when omitted.
            metadata: Optional metadata for the merged run. The merge always records
                ``merged_from_model_run_ids`` alongside whatever you pass.

        Returns:
            Dict describing the newly created (still-populating) run::

                {
                    "model_run_id": str,        # the new run, usable once the job finishes
                    "dataset_ids": List[str],   # datasets the new run spans
                    "job": AsyncJob,            # copy progress; poll or sleep_until_complete()
                }

            The copy's counts (``predictions_copied``, ``predictions_ignored``,
            ``annotation_ids_rewritten``, errors) are reported on the job, not here.
        """
        unique_model_run_ids = list(dict.fromkeys(model_run_ids))
        if len(unique_model_run_ids) < 2:
            raise ValueError(
                "merge_model_runs needs at least two distinct model run ids, got "
                f"{sorted(unique_model_run_ids)}"
            )
        payload: Dict[str, Any] = {
            MODEL_RUN_IDS_KEY: unique_model_run_ids,
        }
        if name is not None:
            payload[NAME_KEY] = name
        if metadata is not None:
            payload[METADATA_KEY] = metadata
        response = self.make_request(payload, "modelRun/merge")
        # The merge endpoint responds 202 with only {job_id, model_run_id,
        # dataset_ids} — not a full job payload — so fetch the job by id rather
        # than parsing it out of this response (the convention used by
        # create_benchmark / add_benchmark_items).
        return {
            MODEL_RUN_ID_KEY: response[MODEL_RUN_ID_KEY],
            DATASET_IDS_KEY: response[DATASET_IDS_KEY],
            "job": self.get_job(response[JOB_ID_KEY]),
        }

    def create_dataset_from_project(
        self,
        project_id: str,
        last_n_tasks: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Dataset:
        """Create a new dataset from an existing Scale or Rapid project.

        If you already have Annotation, SegmentAnnotation, VideoAnnotation,
        Categorization, PolygonAnnotation, ImageAnnotation, DocumentTranscription,
        LidarLinking, LidarAnnotation, or VideoboxAnnotation projects with Scale,
        use this endpoint to import your project directly into Nucleus.

        This endpoint is asynchronous because there can be delays when the
        number of tasks is larger than 1000. As a result, the endpoint returns
        an instance of :class:`AsyncJob`.

        Parameters:
            project_id: The ID of the Scale/Rapid project (retrievable from URL).
            last_n_tasks: If supplied, only pull in this number of the most recent
              tasks. By default the endpoint will pull in all eligible tasks.
            name: The name for your new Nucleus dataset. By default the endpoint
              will use the project's name.

        Returns:
            :class:`Dataset`: The newly created Nucleus dataset as an object.
        """
        payload = {"project_id": project_id}
        if last_n_tasks:
            payload["last_n_tasks"] = str(last_n_tasks)
        if name:
            payload["name"] = name
        response = self.make_request(payload, "dataset/create_from_project")
        return Dataset(response[DATASET_ID_KEY], self)

    def create_dataset(
        self,
        name: str,
        is_scene: Optional[bool] = None,
        use_privacy_mode: bool = False,
        item_metadata_schema: Optional[Dict] = None,
        annotation_metadata_schema: Optional[Dict] = None,
    ) -> Dataset:
        """
        Creates a new, empty dataset.

        Make sure that the dataset is created for the data type you would like to support.
        Be sure to set the ``is_scene`` parameter correctly.

        Parameters:
            name: A human-readable name for the dataset.
            is_scene: Whether the dataset contains strictly :class:`scenes
              <LidarScene>` or :class:`items <DatasetItem>`. This value is immutable.
              Default is False (dataset of items).
            use_privacy_mode: Whether the images of this dataset should be uploaded to Scale. If set to True,
              customer will have to adjust their file access policy with Scale.
            item_metadata_schema: Dict defining item-level metadata schema, structured as::

                {
                    "field_name": {
                        "type": "category" | "number" | "text" | "json"
                        "choices": List[str] | None
                        "description": str | None
                    },
                    ...
                }

            annotation_metadata_schema: Dict defining annotation-level metadata schema.
              Same format as ``item_metadata_schema``.

        Returns:
            :class:`Dataset`: The newly created Nucleus dataset as an object.
        """
        if is_scene is None:
            warnings.warn(
                "The default create_dataset('dataset_name', ...) method without the is_scene parameter will be "
                "deprecated soon in favor of providing the is_scene parameter explicitly. "
                "Please make sure to create a dataset with either create_dataset('dataset_name', is_scene=False, ...) "
                "to upload DatasetItems or create_dataset('dataset_name', is_scene=True, ...) to upload LidarScenes.",
                DeprecationWarning,
            )
            is_scene = False
        response = self.make_request(
            {
                NAME_KEY: name,
                DATASET_IS_SCENE_KEY: is_scene,
                DATASET_PRIVACY_MODE_KEY: use_privacy_mode,
                ANNOTATION_METADATA_SCHEMA_KEY: annotation_metadata_schema,
                ITEM_METADATA_SCHEMA_KEY: item_metadata_schema,
            },
            "dataset/create",
        )
        return Dataset(
            response[DATASET_ID_KEY],
            self,
            name=name,
            is_scene=is_scene,
            use_privacy_mode=use_privacy_mode,
        )

    def delete_dataset(self, dataset_id: str) -> dict:
        """
        Deletes a dataset by ID.

        All items, annotations, and predictions associated with the dataset will
        be deleted as well. Note that if this dataset is linked to a Scale or Rapid
        labeling project, the project itself will not be deleted.

        Parameters:
            dataset_id: The ID of the dataset to delete.

        Returns:
            Payload to indicate deletion invocation.
        """
        return self.make_request({}, f"dataset/{dataset_id}", requests.delete)

    @deprecated("Use Dataset.delete_item instead.")
    def delete_dataset_item(self, dataset_id: str, reference_id) -> dict:
        dataset = self.get_dataset(dataset_id)
        return dataset.delete_item(reference_id)

    @deprecated(msg="Use Dataset.ingest_tasks instead")
    def ingest_tasks(self, dataset_id: str, payload: dict):
        dataset = self.get_dataset(dataset_id)
        return dataset.ingest_tasks(payload["tasks"])

    @deprecated(msg="Use client.create_model instead.")
    def add_model(
        self, name: str, reference_id: str, metadata: Optional[Dict] = None
    ) -> Model:
        return self.create_model(name, reference_id, metadata)

    def create_model(
        self,
        name: str,
        reference_id: str,
        metadata: Optional[Dict] = None,
        bundle_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        trained_slice_ids: Optional[List[str]] = None,
    ) -> Model:
        """Adds a :class:`Model` to Nucleus.

        Parameters:
            name: A human-readable name for the model.
            reference_id: Unique, user-controlled ID for the model. This can be
              used, for example, to link to an external storage of models which
              may have its own id scheme.
            metadata: An arbitrary dictionary of additional data about this model
              that can be stored and retrieved. For example, you can store information
              about the hyperparameters used in training this model.
            bundle_name: Optional name of bundle attached to this model
            tags: Optional list of tags to attach to this model

        Returns:
            :class:`Model`: The newly created model as an object.
        """
        response = self.make_request(
            construct_model_creation_payload(
                name,
                reference_id,
                metadata,
                bundle_name,
                tags,
                trained_slice_ids,
            ),
            "models/add",
        )
        model_id = response.get("model_id", None)
        if not model_id:
            raise ModelCreationError(response.get("error"))

        return Model(
            model_id=model_id,
            name=name,
            reference_id=reference_id,
            metadata=metadata,
            bundle_name=bundle_name,
            client=self,
            tags=tags,
            trained_slice_ids=trained_slice_ids,
        )

    def create_launch_model(
        self,
        name: str,
        reference_id: str,
        bundle_args: Dict[str, Any],
        metadata: Optional[Dict] = None,
        trained_slice_ids: Optional[List[str]] = None,
    ) -> Model:
        """
        Adds a :class:`Model` to Nucleus, as well as a Launch bundle from a given function.

        Parameters:
            name: A human-readable name for the model.
            reference_id: Unique, user-controlled ID for the model. This can be
              used, for example, to link to an external storage of models which
              may have its own ID scheme.
            bundle_args: Dict of kwargs for creating a Launch bundle. See the
              note below for supported keys.
            metadata: An arbitrary dictionary of additional data about this model
              that can be stored and retrieved. For example, you can store information
              about the hyperparameters used in training this model.

        Returns:
            :class:`Model`: The newly created model as an object.

        .. note::

            A bundle consists of exactly ``{predict_fn_or_cls}``,
            ``{load_predict_fn + model}``, or
            ``{load_predict_fn + load_model_fn}``. The exact keys depend on
            the Launch client version (use ``env_params`` for v0.x, or
            ``pytorch_image_tag``/``tensorflow_version`` otherwise).

            Supported ``bundle_args`` keys:

            - ``model_bundle_name``: Unique identifier for the bundle.
            - ``predict_fn_or_cls``: End-to-end callable for inference.
            - ``model``: Trained neural network, e.g. a PyTorch module.
            - ``load_predict_fn``: Returns an inference function given a model.
            - ``load_model_fn``: Loads a model.
            - ``bundle_url``: Self-hosted mode only. Desired bundle location.
            - ``requirements``: List of pip packages.
            - ``app_config``: YAML dict or local path.
            - ``env_params``: Launch v0 framework/CUDA config.
            - ``globals_copy``: Global symbol table (from ``globals()``).
            - ``pytorch_image_tag``: Launch v1 + PyTorch image tag.
            - ``tensorflow_version``: Launch v1 + TensorFlow version.
        """
        from launch import LaunchClient

        launch_client = LaunchClient(api_key=self.api_key)

        model_exists = any(model.name == name for model in self.list_models())
        bundle_exists = any(
            bundle.name == name + "-nucleus-autogen"
            for bundle in launch_client.list_model_bundles()
        )

        if bundle_exists or model_exists:
            raise ModelCreationError(
                "Bundle with the given name already exists, please try a different name"
            )

        kwargs = {
            "model_bundle_name": name + "-nucleus-autogen",
            **bundle_args,
        }
        if hasattr(launch_client, "create_model_bundle_from_callable_v2"):
            # Launch client is >= 1.0.0
            bundle = launch_client.create_model_bundle_from_callable_v2(
                **kwargs
            )
            bundle_name = (
                bundle.name
            )  # both v0 and v1 have a .name field but are different types
        else:
            bundle = launch_client.create_model_bundle(**kwargs)
            bundle_name = bundle.name
        return self.create_model(
            name,
            reference_id,
            metadata,
            bundle_name,
            trained_slice_ids=trained_slice_ids,
        )

    def create_launch_model_from_dir(
        self,
        name: str,
        reference_id: str,
        bundle_from_dir_args: Dict[str, Any],
        metadata: Optional[Dict] = None,
        trained_slice_ids: Optional[List[str]] = None,
    ) -> Model:
        """Adds a :class:`Model` to Nucleus, as well as a Launch bundle from a directory.

        Parameters:
            name: A human-readable name for the model.
            reference_id: Unique, user-controlled ID for the model. This can be
              used, for example, to link to an external storage of models which
              may have its own id scheme.
            bundle_from_dir_args: Dict of kwargs for creating a bundle from
              local directories. See the note below for supported keys.
            metadata: An arbitrary dictionary of additional data about this model
              that can be stored and retrieved. For example, you can store information
              about the hyperparameters used in training this model.

        Returns:
            :class:`Model`: The newly created model as an object.

        .. note::

            Code from one or more local filesystem folders is packaged into a
            zip and uploaded to Scale Launch. Contents are unzipped relative to
            the server-side ``PYTHONPATH``, so module paths should reflect the
            directory structure (e.g. ``my_module.my_file.f``). The exact keys
            depend on the Launch client version (use ``env_params`` for v0.x,
            or ``pytorch_image_tag``/``tensorflow_version`` otherwise).

            Supported ``bundle_from_dir_args`` keys:

            - ``model_bundle_name``: Unique identifier for the bundle.
            - ``base_paths``: Local dirs containing the bundle code.
            - ``requirements_path``: Path to a ``requirements.txt`` file.
            - ``env_params``: Launch v0 framework/CUDA config.
            - ``load_predict_fn_module_path``: Module path for inference fn.
            - ``load_model_fn_module_path``: Module path for model loader.
            - ``app_config``: YAML dict or local path.
            - ``pytorch_image_tag``: Launch v1 + PyTorch image tag.
            - ``tensorflow_version``: Launch v1 + TensorFlow version.

        .. note::

            For example, given this directory structure::

                my_root/
                    my_module1/
                        __init__.py
                        ...files and directories
                        my_inference_file.py
                    my_module2/
                        __init__.py
                        ...files and directories

            Calling with ``base_paths=["my_module1", "my_module2"]`` creates a
            zip without the root directory. Contents are unzipped relative to
            the server-side ``PYTHONPATH``. If ``my_inference_file.py`` has
            ``def f(...)`` as the inference loading function, then
            ``load_predict_fn_module_path`` should be
            ``my_module1.my_inference_file.f``.
        """
        from launch import LaunchClient

        launch_client = LaunchClient(api_key=self.api_key)

        model_exists = any(model.name == name for model in self.list_models())
        bundle_exists = any(
            bundle.name == name + "-nucleus-autogen"
            for bundle in launch_client.list_model_bundles()
        )

        if bundle_exists or model_exists:
            raise ModelCreationError(
                "Bundle with the given name already exists, please try a different name"
            )

        kwargs = {
            "model_bundle_name": name + "-nucleus-autogen",
            **bundle_from_dir_args,
        }

        if hasattr(launch_client, "create_model_bundle_from_dirs_v2"):
            # Launch client is >= 1.0.0, use new fn
            bundle = launch_client.create_model_bundle_from_dirs_v2(**kwargs)
            # Different code paths give different types for bundle, although both have a .name field
            bundle_name = bundle.name
        else:
            # Launch client is < 1.0.0
            bundle = launch_client.create_model_bundle_from_dirs(**kwargs)
            bundle_name = bundle.name

        return self.create_model(
            name,
            reference_id,
            metadata,
            bundle_name,
            trained_slice_ids=trained_slice_ids,
        )

    @deprecated(
        "Model runs have been deprecated and will be removed. Use a Model instead"
    )
    def create_model_run(self, dataset_id: str, payload: dict) -> ModelRun:
        response = self.make_request(
            payload, f"dataset/{dataset_id}/modelRun/create"
        )
        if response.get(STATUS_CODE_KEY, None):
            raise ModelRunCreationError(response.get("error"))

        return ModelRun(
            response[MODEL_RUN_ID_KEY], dataset_id=dataset_id, client=self
        )

    @deprecated(
        "Model runs have been deprecated and will be removed. Use a Model instead."
    )
    def commit_model_run(
        self, model_run_id: str, payload: Optional[dict] = None
    ):
        # TODO: deprecate ModelRun. this should be renamed to calculate_evaluation_metrics
        #   or completely removed in favor of Model class methods
        if payload is None:
            payload = {}
        return self.make_request(payload, f"modelRun/{model_run_id}/commit")

    def get_evaluation_v2(self, evaluation_id: str) -> EvaluationV2:
        """Get an evaluation by id.

        Parameters:
            evaluation_id: Evaluation id (``evalv2_*``).

        Returns:
            :class:`EvaluationV2`.
        """
        data = self.get(f"evaluationsV2/{evaluation_id}")
        return EvaluationV2.from_json(data, self)

    def list_evaluations_v2(self, model_run_id: str) -> List[EvaluationV2]:
        """List evaluations for a model run (newest first).

        Parameters:
            model_run_id: Model run id (``run_*``).

        Returns:
            List of :class:`EvaluationV2`.
        """
        rows = self.get(f"modelRun/{model_run_id}/evaluationsV2")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected list evaluations V2 response: {rows!r}"
            )
        return [EvaluationV2.from_json(r, self) for r in rows]

    def list_evaluation_v2_presets(self) -> List[EvaluationV2Preset]:
        """List the current user's saved Evaluation V2 presets.

        Returns:
            List of :class:`EvaluationV2Preset` (presets are private per user).
        """
        rows = self.get("evaluationV2Presets")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected list evaluation V2 presets response: {rows!r}"
            )
        return [EvaluationV2Preset.from_json(r, self) for r in rows]

    def create_evaluation_v2_preset(
        self,
        name: str,
        *,
        rollup_groups: Optional[List[RollupGroup]] = None,
        allowed_label_matches: Optional[List[AllowedLabelMatch]] = None,
        exclusion_rules: Optional[
            List[Union[EvaluationV2ExclusionRule, Dict[str, Any]]]
        ] = None,
    ) -> EvaluationV2Preset:
        """Create a saved Evaluation V2 preset.

        Parameters:
            name: Preset name. Must be non-empty and unique among the user's
                presets.
            rollup_groups: Optional rollup classes (the primary label
                configuration); each :class:`RollupGroup` maps raw labels onto
                one class name. Mutually exclusive with
                ``allowed_label_matches``.
            allowed_label_matches: Optional legacy label pairs to treat as
                matches. Prefer ``rollup_groups``.
            exclusion_rules: Optional rules that drop items/annotations (same
                types accepted by :meth:`create_benchmark_evaluation_v2`).

        Returns:
            :class:`EvaluationV2Preset`: The created preset.
        """
        if rollup_groups is not None and allowed_label_matches is not None:
            raise ValueError(
                "rollup_groups and allowed_label_matches cannot both be set"
            )
        payload: Dict[str, Any] = {NAME_KEY: name}
        if rollup_groups is not None:
            payload[ROLLUP_GROUPS_CAMEL_KEY] = [
                g.to_api_dict() for g in rollup_groups
            ]
        if allowed_label_matches is not None:
            payload[ALLOWED_LABEL_MATCHES_CAMEL_KEY] = [
                m.to_api_dict() for m in allowed_label_matches
            ]
        if exclusion_rules is not None:
            payload[EXCLUSION_RULES_CAMEL_KEY] = [
                rule.to_api_dict() if hasattr(rule, "to_api_dict") else rule
                for rule in exclusion_rules
            ]
        data = self.post(payload, "evaluationV2Presets")
        return EvaluationV2Preset.from_json(data, self)

    def update_evaluation_v2_preset(
        self,
        preset_id: str,
        *,
        name: Any = _UNSET,
        rollup_groups: Any = _UNSET,
        allowed_label_matches: Any = _UNSET,
        exclusion_rules: Any = _UNSET,
    ) -> EvaluationV2Preset:
        """Update a saved Evaluation V2 preset.

        Only the fields you pass are changed. Passing ``rollup_groups=None``
        or ``exclusion_rules=None`` clears that field; omitting an argument
        leaves it unchanged.

        Parameters:
            preset_id: Preset id (``prev_*``). Must be owned by the caller.
            name: Optional new name.
            rollup_groups: Optional new rollup classes, or ``None`` to clear.
                Mutually exclusive with ``allowed_label_matches``.
            allowed_label_matches: Optional new legacy label-match list.
            exclusion_rules: Optional new exclusion rules, or ``None`` to clear.

        Returns:
            :class:`EvaluationV2Preset`: The updated preset.
        """
        if (
            rollup_groups is not _UNSET
            and rollup_groups is not None
            and allowed_label_matches is not _UNSET
            and allowed_label_matches is not None
        ):
            raise ValueError(
                "rollup_groups and allowed_label_matches cannot both be set"
            )
        payload: Dict[str, Any] = {}
        if name is not _UNSET:
            payload[NAME_KEY] = name
        if rollup_groups is not _UNSET:
            payload[ROLLUP_GROUPS_CAMEL_KEY] = (
                None
                if rollup_groups is None
                else [g.to_api_dict() for g in rollup_groups]
            )
        if allowed_label_matches is not _UNSET:
            payload[ALLOWED_LABEL_MATCHES_CAMEL_KEY] = (
                None
                if allowed_label_matches is None
                else [m.to_api_dict() for m in allowed_label_matches]
            )
        if exclusion_rules is not _UNSET:
            payload[EXCLUSION_RULES_CAMEL_KEY] = (
                None
                if exclusion_rules is None
                else [
                    (
                        rule.to_api_dict()
                        if hasattr(rule, "to_api_dict")
                        else rule
                    )
                    for rule in exclusion_rules
                ]
            )
        data = self.patch(payload, f"evaluationV2Presets/{preset_id}")
        return EvaluationV2Preset.from_json(data, self)

    def delete_evaluation_v2_preset(self, preset_id: str) -> None:
        """Delete a saved Evaluation V2 preset.

        Parameters:
            preset_id: Preset id (``prev_*``). Must be owned by the caller.
        """
        self.make_request(
            {},
            f"evaluationV2Presets/{preset_id}",
            requests_command=requests.delete,
            return_raw_response=True,
        )

    def create_benchmark(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        parent_benchmark_id: Optional[str] = None,
        bump_type: Optional[str] = None,
        version_major: Optional[int] = None,
        version_minor: Optional[int] = None,
        version_label: Optional[str] = None,
        removed_item_ids: Optional[List[str]] = None,
        draft: bool = False,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> Benchmark:
        """Create a benchmark from ground-truth items.

        Provide members through any combination of sources: explicit
        ``item_ids``, ``(dataset_id, ref_id)`` pairs via ``items``, one or more
        slices via ``slice_id`` / ``slice_ids``, and one or more datasets via
        ``dataset_id`` / ``dataset_ids``. Members are unioned and de-duplicated
        across all sources; at least one source is required. Items without
        ground truth are skipped. Membership is frozen at creation.

        Creation is **asynchronous**: the server creates the benchmark in a
        ``"building"`` state and streams its members in via a background job.
        By default this method blocks until that job finishes and returns the
        completed (``"ready"``) benchmark. Pass ``wait_for_completion=False``
        to return immediately with a ``"building"`` benchmark you can poll via
        :meth:`Benchmark.refresh` (checking ``benchmark.status``).

        **Versioning.** Pass ``parent_benchmark_id`` to create a new *version*
        downstream of an existing benchmark: the child inherits the parent's
        items, the source arguments **add** on top, and ``removed_item_ids``
        **prune** inherited items (``final set = parent ∪ added ∖ removed``).
        The version defaults to a minor bump; pass ``bump_type="major"`` for a
        major bump, or an explicit ``version_major`` + ``version_minor`` (which
        must exceed the parent's). ``parent_benchmark_id`` alone is a valid
        source (a pure re-version). Lineage is immutable — a parent is fixed at
        creation.

        **Drafts.** Pass ``draft=True`` to create a mutable draft instead of
        building in one shot. Sources are then optional (an empty draft is
        valid); add items later with :meth:`Benchmark.add_items`, remove with
        :meth:`Benchmark.remove_items`, then :meth:`Benchmark.finalize`. A draft
        cannot be evaluated until finalized.

        Parameters:
            name: Benchmark display name.
            description: Optional description.
            metadata: Optional arbitrary metadata dict.
            item_ids: Global dataset item ids (``di_*``).
            items: ``{"dataset_id": ..., "ref_id": ...}`` pairs.
            slice_id: Slice id (``slc_*``) whose items become members.
            dataset_id: Dataset id (``ds_*``) whose items become members.
            slice_ids: Multiple slice ids whose items become members.
            dataset_ids: Multiple dataset ids whose items become members.
            parent_benchmark_id: Create as a new version downstream of this
                benchmark, inheriting its items.
            bump_type: ``"minor"`` (default) or ``"major"`` version bump
                relative to the parent. Ignored without ``parent_benchmark_id``.
            version_major: Explicit major version (with ``version_minor``); must
                exceed the parent's version.
            version_minor: Explicit minor version (with ``version_major``).
            version_label: Optional human-readable version label.
            removed_item_ids: Inherited item ids (``di_*``) to prune from the
                parent's set. Only valid with ``parent_benchmark_id``.
            draft: Create a mutable draft (sources optional) instead of a
                one-shot build.
            wait_for_completion: Block until the build/seed job finishes and
                return the resulting benchmark (default). If ``False``, return
                immediately. Ignored for an empty draft (no job is started).
            verbose: Log build-job polling progress while waiting.

        Returns:
            :class:`Benchmark`: The created benchmark — ``"ready"`` for a
            completed one-shot build, ``"draft"`` for a draft, otherwise
            ``"building"``.
        """
        has_source = any(
            source
            for source in (
                item_ids,
                items,
                slice_id,
                dataset_id,
                slice_ids,
                dataset_ids,
                parent_benchmark_id,
            )
        )
        # A draft may start empty (items added later); a version inherits from
        # its parent. Otherwise at least one explicit source is required.
        if not has_source and not draft:
            raise ValueError(
                "Provide at least one of item_ids, items, slice_id(s), "
                "dataset_id(s), or parent_benchmark_id to define benchmark "
                "membership (or pass draft=True to start an empty draft)"
            )
        if removed_item_ids is not None and parent_benchmark_id is None:
            raise ValueError(
                "removed_item_ids is only valid together with "
                "parent_benchmark_id"
            )
        payload: Dict[str, Any] = {NAME_KEY: name}
        optional_fields = {
            DESCRIPTION_KEY: description,
            METADATA_KEY: metadata,
            ITEM_IDS_KEY: item_ids,
            ITEMS_KEY: items,
            SLICE_ID_KEY: slice_id,
            DATASET_ID_KEY: dataset_id,
            SLICE_IDS_KEY: slice_ids,
            DATASET_IDS_KEY: dataset_ids,
            PARENT_BENCHMARK_ID_KEY: parent_benchmark_id,
            BUMP_TYPE_KEY: bump_type,
            VERSION_MAJOR_KEY: version_major,
            VERSION_MINOR_KEY: version_minor,
            VERSION_LABEL_KEY: version_label,
            REMOVED_ITEM_IDS_KEY: removed_item_ids,
        }
        payload.update(
            {
                key: value
                for key, value in optional_fields.items()
                if value is not None
            }
        )
        if draft:
            payload[DRAFT_KEY] = True

        # Async: the server responds 202 with {benchmark_id, job_id}. The
        # benchmark row already exists (in 'building', or 'draft'); the build /
        # seed job streams members in. An empty draft starts no job, so job_id
        # may be null there.
        response = self.post(payload, "benchmarks")
        benchmark_id = response[BENCHMARK_ID_KEY]
        job_id = response.get(JOB_ID_KEY)
        if wait_for_completion and job_id is not None:
            self.get_job(job_id).sleep_until_complete(verbose_std_out=verbose)
        elif wait_for_completion and job_id is None and not draft:
            raise ValueError(
                "Server did not return a job_id in the create-benchmark "
                "response; cannot poll for completion. Pass "
                "wait_for_completion=False to suppress this error."
            )

        return self.get_benchmark(benchmark_id)

    def list_benchmarks(self) -> List[Benchmark]:
        """List benchmarks visible to the current user.

        Returns:
            List of :class:`Benchmark`.
        """
        rows = self.get("benchmarks")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected list benchmarks response: {rows!r}"
            )
        return [Benchmark.from_json(r, self) for r in rows]

    def get_benchmark(self, benchmark_id: str) -> Benchmark:
        """Get a benchmark by id.

        Parameters:
            benchmark_id: Benchmark id (``bm_*``).

        Returns:
            :class:`Benchmark`.
        """
        data = self.get(f"benchmarks/{benchmark_id}")
        return Benchmark.from_json(data, self)

    def update_benchmark(
        self,
        benchmark_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Benchmark:
        """Update a benchmark's name, description, or metadata.

        Only the arguments you pass are changed. Benchmark membership is
        frozen at creation and cannot be updated.

        Parameters:
            benchmark_id: Benchmark id (``bm_*``).
            name: Optional new display name.
            description: Optional new description.
            metadata: Optional new metadata dict.

        Returns:
            :class:`Benchmark`: The updated benchmark.
        """
        payload: Dict[str, Any] = {}
        if name is not None:
            payload[NAME_KEY] = name
        if description is not None:
            payload[DESCRIPTION_KEY] = description
        if metadata is not None:
            payload[METADATA_KEY] = metadata
        data = self.patch(payload, f"benchmarks/{benchmark_id}")
        return Benchmark.from_json(data, self)

    def delete_benchmark(self, benchmark_id: str) -> None:
        """Delete a benchmark.

        Parameters:
            benchmark_id: Benchmark id (``bm_*``).
        """
        self.make_request(
            {},
            f"benchmarks/{benchmark_id}",
            requests_command=requests.delete,
            return_raw_response=True,
        )

    def list_benchmark_items(
        self,
        benchmark_id: str,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> BenchmarkItemsPage:
        """Return one page of a benchmark's member item ids.

        Parameters:
            benchmark_id: Benchmark id (``bm_*``).
            limit: Optional page size.
            offset: Optional offset for pagination.

        Returns:
            :class:`~nucleus.data_transfer_object.evaluation_v2.BenchmarkItemsPage`.
        """
        route = f"benchmarks/{benchmark_id}/items"
        params = []
        if limit is not None:
            params.append(f"limit={limit}")
        if offset is not None:
            params.append(f"offset={offset}")
        if params:
            route = f"{route}?{'&'.join(params)}"
        data = self.get(route)
        return BenchmarkItemsPage.parse_obj(data)

    def add_benchmark_items(
        self,
        benchmark_id: str,
        *,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        scene_ids: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> None:
        """Add items to a **draft** benchmark.

        Only valid while the benchmark is a draft (``status == "draft"``); a
        finalized benchmark is immutable (409 otherwise). Accepts the same
        sources as :meth:`create_benchmark`; members are unioned/de-duplicated
        and items without ground truth are skipped.

        Like create, this is **asynchronous**: the server streams the sources in
        via a background job. By default this blocks until the job finishes.

        Parameters:
            benchmark_id: Draft benchmark id (``bm_*``).
            item_ids: Global dataset item ids (``di_*``).
            items: ``{"dataset_id": ..., "ref_id": ...}`` pairs.
            slice_id: Slice id whose items are added.
            dataset_id: Dataset id whose items are added.
            slice_ids: Multiple slice ids whose items are added.
            dataset_ids: Multiple dataset ids whose items are added.
            scene_ids: Scene ids (``scn_*``) whose items are added.
            wait_for_completion: Block until the add job finishes (default).
            verbose: Log add-job polling progress while waiting.
        """
        has_source = any(
            source
            for source in (
                item_ids,
                items,
                slice_id,
                dataset_id,
                slice_ids,
                dataset_ids,
                scene_ids,
            )
        )
        if not has_source:
            raise ValueError(
                "Provide at least one of item_ids, items, slice_id(s), "
                "dataset_id(s), or scene_ids to add"
            )
        payload: Dict[str, Any] = {}
        if item_ids is not None:
            payload[ITEM_IDS_KEY] = item_ids
        if items is not None:
            payload[ITEMS_KEY] = items
        if slice_id is not None:
            payload[SLICE_ID_KEY] = slice_id
        if dataset_id is not None:
            payload[DATASET_ID_KEY] = dataset_id
        if slice_ids is not None:
            payload[SLICE_IDS_KEY] = slice_ids
        if dataset_ids is not None:
            payload[DATASET_IDS_KEY] = dataset_ids
        if scene_ids is not None:
            payload[SCENE_IDS_KEY] = scene_ids

        # 202 with {job_id}; the append job streams items into the draft and
        # leaves its status 'draft'.
        response = self.post(payload, f"benchmarks/{benchmark_id}/items")
        job_id = response.get(JOB_ID_KEY)
        if wait_for_completion:
            if job_id is None:
                raise ValueError(
                    "Server did not return a job_id in the add-benchmark-items "
                    "response; cannot poll for completion. Pass "
                    "wait_for_completion=False to suppress this error."
                )
            self.get_job(job_id).sleep_until_complete(verbose_std_out=verbose)

    def remove_benchmark_items(
        self, benchmark_id: str, item_ids: List[str]
    ) -> None:
        """Remove items from a **draft** benchmark (synchronous).

        Only valid while the benchmark is a draft (409 otherwise). Unknown ids
        are ignored.

        Parameters:
            benchmark_id: Draft benchmark id (``bm_*``).
            item_ids: Dataset item ids (``di_*``) to remove.
        """
        self.make_request(
            {ITEM_IDS_KEY: item_ids},
            f"benchmarks/{benchmark_id}/items",
            requests_command=requests.delete,
            return_raw_response=True,
        )

    def finalize_benchmark(self, benchmark_id: str) -> Benchmark:
        """Finalize a **draft** benchmark, freezing it into a ``"ready"`` one.

        After finalizing, the benchmark is immutable and can be evaluated. Fails
        (409) if it is not a draft or an add-items job is still in flight, and
        (400) if the draft is empty.

        Parameters:
            benchmark_id: Draft benchmark id (``bm_*``).

        Returns:
            :class:`Benchmark`: The finalized (``"ready"``) benchmark.
        """
        data = self.post({}, f"benchmarks/{benchmark_id}/finalize")
        return Benchmark.from_json(data, self)

    # --------------------------------------------------------------------- #
    # Training sets
    # --------------------------------------------------------------------- #
    @staticmethod
    def _training_set_source_payload(
        *,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        training_set_ids: Optional[List[str]] = None,
        scene_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Map each non-None membership source to its payload key.

        Shared by create / add / new-version. Members are unioned and
        de-duplicated across every source by the backend.
        """
        source_fields = {
            ITEM_IDS_KEY: item_ids,
            ITEMS_KEY: items,
            SLICE_ID_KEY: slice_id,
            DATASET_ID_KEY: dataset_id,
            SLICE_IDS_KEY: slice_ids,
            DATASET_IDS_KEY: dataset_ids,
            TRAINING_SET_IDS_KEY: training_set_ids,
            SCENE_IDS_KEY: scene_ids,
        }
        return {
            key: value
            for key, value in source_fields.items()
            if value is not None
        }

    def create_training_set(
        self,
        name: str,
        *,
        model: Union[Model, str],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        training_set_ids: Optional[List[str]] = None,
        parent_training_set_id: Optional[str] = None,
        bump_type: Optional[str] = None,
        version_major: Optional[int] = None,
        version_minor: Optional[int] = None,
        version_label: Optional[str] = None,
        removed_item_ids: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> TrainingSet:
        """Create a training set scoped to a model and attach it.

        A training set is a mutable, versioned collection of ``dataset_item``
        ids spanning one or more datasets. Provide members through any
        combination of sources: explicit ``item_ids``, ``(dataset_id,
        reference_id)`` pairs via ``items``, one or more slices via ``slice_id``
        / ``slice_ids``, one or more datasets via ``dataset_id`` /
        ``dataset_ids``, and the members of other training sets via
        ``training_set_ids``. Members are unioned and de-duplicated; at least
        one source is required.

        Creation is **asynchronous**: the server creates the training set in a
        ``"building"`` state and streams its members in via a background job.
        By default this blocks until that job finishes and returns the
        ``"ready"`` training set. Pass ``wait_for_completion=False`` to return
        immediately with a ``"building"`` set you can poll via
        :meth:`TrainingSet.refresh`.

        **Versioning.** Pass ``parent_training_set_id`` to create a new version
        downstream of an existing training set: the child inherits the parent's
        items, the source arguments **add** on top, and ``removed_item_ids``
        **prune** inherited items (``final set = parent ∪ added ∖ removed``).
        The version defaults to a minor bump; pass ``bump_type="major"`` or an
        explicit ``version_major`` + ``version_minor``.

        Parameters:
            name: Training set display name.
            model: The :class:`Model` (or model id) to scope and attach to.
            description: Optional description.
            metadata: Optional arbitrary metadata dict.
            item_ids: Global dataset item ids (``di_*``).
            items: ``{"dataset_id": ..., "reference_id": ...}`` pairs.
            slice_id: Slice id (``slc_*``) whose items become members.
            dataset_id: Dataset id (``ds_*``) whose items become members.
            slice_ids: Multiple slice ids whose items become members.
            dataset_ids: Multiple dataset ids whose items become members.
            training_set_ids: Other training set ids whose members are unioned in.
            parent_training_set_id: Create as a new version downstream of this
                training set, inheriting its items.
            bump_type: ``"minor"`` (default) or ``"major"`` version bump relative
                to the parent. Ignored without ``parent_training_set_id``.
            version_major: Explicit major version (with ``version_minor``).
            version_minor: Explicit minor version (with ``version_major``).
            version_label: Optional human-readable version label.
            removed_item_ids: Inherited item ids (``di_*``) to prune from the
                parent's set. Only valid with ``parent_training_set_id``.
            wait_for_completion: Block until the build job finishes and return
                the resulting training set (default).
            verbose: Log build-job polling progress while waiting.

        Returns:
            :class:`TrainingSet`: The created training set.
        """
        model_id = model.id if isinstance(model, Model) else model
        sources = self._training_set_source_payload(
            item_ids=item_ids,
            items=items,
            slice_id=slice_id,
            dataset_id=dataset_id,
            slice_ids=slice_ids,
            dataset_ids=dataset_ids,
            training_set_ids=training_set_ids,
        )
        if not any(sources.values()) and not parent_training_set_id:
            raise ValueError(
                "Provide at least one of item_ids, items, slice_id(s), "
                "dataset_id(s), training_set_ids, or parent_training_set_id to "
                "define training set membership"
            )
        if removed_item_ids is not None and parent_training_set_id is None:
            raise ValueError(
                "removed_item_ids is only valid together with "
                "parent_training_set_id"
            )
        payload: Dict[str, Any] = {NAME_KEY: name, **sources}
        version_fields = {
            DESCRIPTION_KEY: description,
            METADATA_KEY: metadata,
            PARENT_TRAINING_SET_ID_KEY: parent_training_set_id,
            BUMP_TYPE_KEY: bump_type,
            VERSION_MAJOR_KEY: version_major,
            VERSION_MINOR_KEY: version_minor,
            VERSION_LABEL_KEY: version_label,
            REMOVED_ITEM_IDS_KEY: removed_item_ids,
        }
        payload.update(
            {
                key: value
                for key, value in version_fields.items()
                if value is not None
            }
        )

        # Async: server responds 202 with {training_set_id, job_id}. The row
        # already exists (in 'building'); the seed job streams members in.
        response = self.post(payload, f"model/{model_id}/trainingSet")
        training_set_id = response[TRAINING_SET_ID_KEY]
        job_id = response.get(JOB_ID_KEY)
        if wait_for_completion and job_id is not None:
            self.get_job(job_id).sleep_until_complete(verbose_std_out=verbose)
        elif wait_for_completion and job_id is None:
            raise ValueError(
                "Server did not return a job_id in the create-training-set "
                "response; cannot poll for completion. Pass "
                "wait_for_completion=False to suppress this error."
            )
        return self.get_training_set(training_set_id)

    def list_training_sets(self) -> List[TrainingSet]:
        """List training sets visible to the current user.

        Returns:
            List of :class:`TrainingSet`.
        """
        rows = self.get("trainingSets")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected list training sets response: {rows!r}"
            )
        return [TrainingSet.from_json(r, self) for r in rows]

    def get_training_set(self, training_set_id: str) -> TrainingSet:
        """Get a training set by id.

        Parameters:
            training_set_id: Training set id.

        Returns:
            :class:`TrainingSet`.
        """
        data = self.get(f"trainingSets/{training_set_id}")
        return TrainingSet.from_json(data, self)

    def get_model_training_set(self, model: Union[Model, str]) -> TrainingSet:
        """Get the training set currently pinned to a model.

        Parameters:
            model: The :class:`Model` (or model id).

        Returns:
            :class:`TrainingSet`: The model's currently pinned training set.
        """
        model_id = model.id if isinstance(model, Model) else model
        data = self.get(f"model/{model_id}/trainingSet")
        return TrainingSet.from_json(data, self)

    def repin_training_set(
        self, model: Union[Model, str], training_set_id: str
    ) -> TrainingSet:
        """Pin a model to a specific training set (version).

        Parameters:
            model: The :class:`Model` (or model id) to repin.
            training_set_id: The training set id to pin the model to.

        Returns:
            :class:`TrainingSet`: The now-pinned training set.
        """
        model_id = model.id if isinstance(model, Model) else model
        data = self.put(
            {TRAINING_SET_ID_KEY: training_set_id},
            f"model/{model_id}/trainingSet",
        )
        return TrainingSet.from_json(data, self)

    def update_training_set(
        self,
        training_set_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrainingSet:
        """Update a training set's name, description, or metadata.

        Only the arguments you pass are changed. Use
        :meth:`add_training_set_items` / :meth:`remove_training_set_items` to
        change membership.

        Parameters:
            training_set_id: Training set id.
            name: Optional new display name.
            description: Optional new description.
            metadata: Optional new metadata dict.

        Returns:
            :class:`TrainingSet`: The updated training set.
        """
        payload: Dict[str, Any] = {}
        if name is not None:
            payload[NAME_KEY] = name
        if description is not None:
            payload[DESCRIPTION_KEY] = description
        if metadata is not None:
            payload[METADATA_KEY] = metadata
        data = self.patch(payload, f"trainingSets/{training_set_id}")
        return TrainingSet.from_json(data, self)

    def delete_training_set(self, training_set_id: str) -> None:
        """Delete a training set.

        Parameters:
            training_set_id: Training set id.
        """
        self.make_request(
            {},
            f"trainingSets/{training_set_id}",
            requests_command=requests.delete,
            return_raw_response=True,
        )

    def list_training_set_items(
        self,
        training_set_id: str,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> TrainingSetItemsPage:
        """Return one page of a training set's member item ids.

        Parameters:
            training_set_id: Training set id.
            limit: Optional page size.
            offset: Optional offset for pagination.

        Returns:
            :class:`~nucleus.data_transfer_object.training_set.TrainingSetItemsPage`.
        """
        route = f"trainingSets/{training_set_id}/items"
        params = []
        if limit is not None:
            params.append(f"limit={limit}")
        if offset is not None:
            params.append(f"offset={offset}")
        if params:
            route = f"{route}?{'&'.join(params)}"
        data = self.get(route)
        return TrainingSetItemsPage.parse_obj(data)

    def _export_training_set_records(
        self,
        training_set_id: str,
        *,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Page the training-set export endpoint, returning the raw records.

        Each record is the backend's export shape (``dataset_item_id``,
        ``dataset_id``, ``reference_id``, ``metadata``, ``image_location``,
        ``pointcloud_location``, ``width``, ``height``). This preserves fields
        (notably ``dataset_id``) that :class:`~nucleus.dataset_item.DatasetItem`
        cannot hold, so file/media exports use these directly.
        """
        accumulated: List[Dict[str, Any]] = []
        offset = 0
        while True:
            route = (
                f"trainingSets/{training_set_id}/export"
                f"?limit={limit}&offset={offset}"
            )
            data = self.get(route)
            items = data.get("items", []) or []
            total = data.get("total", 0)
            accumulated.extend(items)
            # Stop when the server returns an empty page or we've collected the
            # advertised total (guards against an off-by-one final page).
            if not items or len(accumulated) >= total:
                break
            offset += limit
        return accumulated

    @staticmethod
    def _training_set_record_to_dataset_item(
        record: Dict[str, Any],
    ) -> DatasetItem:
        """Hydrate one export record into a :class:`DatasetItem`.

        The export record keys ``image_location`` / ``pointcloud_location`` are
        remapped to the ``image_url`` / ``pointcloud_url`` keys that
        :meth:`DatasetItem.from_json` reads; ``width`` / ``height`` (which
        ``from_json`` does not map) are set afterwards. Note ``dataset_id`` has
        no home on ``DatasetItem`` — use the raw records (e.g. via
        :meth:`TrainingSet.export_to_file`) when you need it.
        """
        adapted = dict(record)
        image_location = record.get(IMAGE_LOCATION_KEY)
        pointcloud_location = record.get(POINTCLOUD_LOCATION_KEY)
        if image_location:
            adapted[IMAGE_URL_KEY] = image_location
        if pointcloud_location:
            adapted[POINTCLOUD_URL_KEY] = pointcloud_location
        item = DatasetItem.from_json(adapted)
        item.width = record.get(WIDTH_KEY)
        item.height = record.get(HEIGHT_KEY)
        return item

    def export_training_set_items(
        self,
        training_set_id: str,
        *,
        limit: int = 1000,
    ) -> List[DatasetItem]:
        """Export a training set's members as fully-hydrated dataset items.

        Pages the export endpoint from ``offset=0`` in ``limit``-sized batches
        until every member has been fetched, converting each record into a
        :class:`~nucleus.dataset_item.DatasetItem` (with ``image_location`` /
        ``pointcloud_location``, ``reference_id``, ``metadata``, ``width`` /
        ``height`` and the server-side ``dataset_item_id``).

        Parameters:
            training_set_id: Training set id.
            limit: Page size for the underlying export requests.

        Returns:
            List[:class:`~nucleus.dataset_item.DatasetItem`]: Every member item.
        """
        records = self._export_training_set_records(
            training_set_id, limit=limit
        )
        return [
            self._training_set_record_to_dataset_item(record)
            for record in records
        ]

    def add_training_set_items(
        self,
        training_set_id: str,
        *,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        training_set_ids: Optional[List[str]] = None,
        scene_ids: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> None:
        """Add items to a training set.

        Accepts the same sources as :meth:`create_training_set`; members are
        unioned/de-duplicated with the existing set.

        Like create, this is **asynchronous**: the server streams the sources in
        via a background job. By default this blocks until the job finishes.

        Parameters:
            training_set_id: Training set id.
            item_ids: Global dataset item ids (``di_*``).
            items: ``{"dataset_id": ..., "reference_id": ...}`` pairs.
            slice_id: Slice id whose items are added.
            dataset_id: Dataset id whose items are added.
            slice_ids: Multiple slice ids whose items are added.
            dataset_ids: Multiple dataset ids whose items are added.
            training_set_ids: Other training set ids whose members are added.
            scene_ids: Scene ids (``scn_*``) whose items are added.
            wait_for_completion: Block until the add job finishes (default).
            verbose: Log add-job polling progress while waiting.
        """
        payload = self._training_set_source_payload(
            item_ids=item_ids,
            items=items,
            slice_id=slice_id,
            dataset_id=dataset_id,
            slice_ids=slice_ids,
            dataset_ids=dataset_ids,
            training_set_ids=training_set_ids,
            scene_ids=scene_ids,
        )
        if not any(payload.values()):
            raise ValueError(
                "Provide at least one of item_ids, items, slice_id(s), "
                "dataset_id(s), training_set_ids, or scene_ids to add"
            )
        # 202 with {job_id}; the append job streams items into the set.
        response = self.post(payload, f"trainingSets/{training_set_id}/items")
        job_id = response.get(JOB_ID_KEY)
        if wait_for_completion:
            if job_id is None:
                raise ValueError(
                    "Server did not return a job_id in the "
                    "add-training-set-items response; cannot poll for "
                    "completion. Pass wait_for_completion=False to suppress "
                    "this error."
                )
            self.get_job(job_id).sleep_until_complete(verbose_std_out=verbose)

    def remove_training_set_items(
        self, training_set_id: str, item_ids: List[str]
    ) -> None:
        """Remove items from a training set (synchronous).

        Unknown ids are ignored.

        Parameters:
            training_set_id: Training set id.
            item_ids: Dataset item ids (``di_*``) to remove.
        """
        self.make_request(
            {ITEM_IDS_KEY: item_ids},
            f"trainingSets/{training_set_id}/items",
            requests_command=requests.delete,
            return_raw_response=True,
        )

    def create_training_set_version(
        self,
        training_set_id: str,
        *,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        training_set_ids: Optional[List[str]] = None,
        removed_item_ids: Optional[List[str]] = None,
        bump_type: Optional[str] = None,
        version_major: Optional[int] = None,
        version_minor: Optional[int] = None,
        version_label: Optional[str] = None,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> TrainingSet:
        """Create a new version downstream of an existing training set.

        The child inherits the parent's items, the source arguments add on top,
        and ``removed_item_ids`` prune inherited items
        (``final set = parent ∪ added ∖ removed``). The version defaults to a
        minor bump; pass ``bump_type="major"`` or explicit ``version_major`` +
        ``version_minor``.

        Like create, this is **asynchronous**: by default it blocks until the
        seed job finishes and returns the new ``"ready"`` version.

        Parameters:
            training_set_id: Parent training set id to version from.
            item_ids: Global dataset item ids (``di_*``) to add on top.
            items: ``{"dataset_id": ..., "reference_id": ...}`` pairs to add.
            slice_id: Slice id whose items are added.
            dataset_id: Dataset id whose items are added.
            slice_ids: Multiple slice ids whose items are added.
            dataset_ids: Multiple dataset ids whose items are added.
            training_set_ids: Other training set ids whose members are added.
            removed_item_ids: Inherited item ids (``di_*``) to prune.
            bump_type: ``"minor"`` (default) or ``"major"`` version bump.
            version_major: Explicit major version (with ``version_minor``).
            version_minor: Explicit minor version (with ``version_major``).
            version_label: Optional human-readable version label.
            wait_for_completion: Block until the seed job finishes (default).
            verbose: Log seed-job polling progress while waiting.

        Returns:
            :class:`TrainingSet`: The newly created version.
        """
        payload = self._training_set_source_payload(
            item_ids=item_ids,
            items=items,
            slice_id=slice_id,
            dataset_id=dataset_id,
            slice_ids=slice_ids,
            dataset_ids=dataset_ids,
            training_set_ids=training_set_ids,
        )
        version_fields = {
            REMOVED_ITEM_IDS_KEY: removed_item_ids,
            BUMP_TYPE_KEY: bump_type,
            VERSION_MAJOR_KEY: version_major,
            VERSION_MINOR_KEY: version_minor,
            VERSION_LABEL_KEY: version_label,
        }
        payload.update(
            {
                key: value
                for key, value in version_fields.items()
                if value is not None
            }
        )
        response = self.post(
            payload, f"trainingSets/{training_set_id}/versions"
        )
        new_id = response[TRAINING_SET_ID_KEY]
        job_id = response.get(JOB_ID_KEY)
        if wait_for_completion and job_id is not None:
            self.get_job(job_id).sleep_until_complete(verbose_std_out=verbose)
        elif wait_for_completion and job_id is None:
            raise ValueError(
                "Server did not return a job_id in the create-training-set-"
                "version response; cannot poll for completion. Pass "
                "wait_for_completion=False to suppress this error."
            )
        return self.get_training_set(new_id)

    def list_training_set_family(
        self, training_set_id: str
    ) -> List[TrainingSet]:
        """Return every version in a training set's lineage (its family).

        Parameters:
            training_set_id: Any training set id in the lineage.

        Returns:
            List of :class:`TrainingSet` sharing the lineage root.
        """
        rows = self.get(f"trainingSets/{training_set_id}/family")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected training set family response: {rows!r}"
            )
        return [TrainingSet.from_json(r, self) for r in rows]

    def create_benchmark_evaluation_v2(
        self,
        benchmark_id: str,
        model_run_id: str,
        *,
        name: Optional[str] = None,
        rollup_groups: Optional[List[RollupGroup]] = None,
        allowed_label_matches: Optional[List[AllowedLabelMatch]] = None,
        allowed_label_matches_id: Optional[str] = None,
        exclusion_rules: Optional[
            List[Union[EvaluationV2ExclusionRule, Dict[str, Any]]]
        ] = None,
        preset: Optional[EvaluationV2Preset] = None,
    ) -> EvaluationV2:
        """Evaluate a model run against a benchmark.

        Every benchmark item is scored: items the model run has no
        predictions for count as false negatives, keeping scores comparable
        across runs with different coverage. The evaluation runs in the
        background — call :meth:`EvaluationV2.wait_for_completion`, then
        :meth:`EvaluationV2.charts` or :meth:`EvaluationV2.examples`.

        The benchmark may span datasets the model run has no predictions in at
        all. Those members are scored as false negatives like any other
        uncovered item, so a partial run still ranks comparably rather than
        being rejected. To give a run predictions across several datasets, use
        :meth:`Dataset.upload_predictions_for_model_run`.

        Parameters:
            benchmark_id: Benchmark id (``bm_*``).
            model_run_id: Model run id (``run_*``). It need not cover the
                benchmark's datasets — coverage may be partial, or empty.
            name: Optional display name.
            rollup_groups: Optional rollup classes (the primary label
                configuration); each :class:`RollupGroup` maps raw labels
                onto one class name. Mutually exclusive with the
                ``allowed_label_matches*`` arguments.
            allowed_label_matches: Optional legacy label pairs to treat as
                matches. Prefer ``rollup_groups``.
            allowed_label_matches_id: Optional id of a saved label-match
                configuration.
            exclusion_rules: Optional rules that drop items/annotations
                before metrics are computed (see
                :mod:`nucleus.evaluation_v2_exclusions`).
            preset: Optional :class:`EvaluationV2Preset` whose label
                configuration and ``exclusion_rules`` seed this evaluation.
                Explicit arguments take precedence over the preset's values.

        Returns:
            :class:`EvaluationV2`: The created evaluation.
        """
        if preset is not None:
            if (
                rollup_groups is None
                and allowed_label_matches is None
                and allowed_label_matches_id is None
            ):
                rollup_groups = preset.rollup_groups
                if rollup_groups is None:
                    allowed_label_matches = preset.allowed_label_matches
            if exclusion_rules is None and preset.exclusion_rules is not None:
                exclusion_rules = list(preset.exclusion_rules)
        label_configs = [
            config
            for config in (
                rollup_groups,
                allowed_label_matches,
                allowed_label_matches_id,
            )
            if config is not None
        ]
        if len(label_configs) > 1:
            raise ValueError(
                "Set at most one of rollup_groups, allowed_label_matches, "
                "or allowed_label_matches_id"
            )
        payload: Dict[str, Any] = {MODEL_RUN_ID_KEY: model_run_id}
        if name is not None:
            payload[NAME_KEY] = name
        if rollup_groups is not None:
            payload[ROLLUP_GROUPS_CAMEL_KEY] = [
                g.to_api_dict() for g in rollup_groups
            ]
        if allowed_label_matches is not None:
            payload["allowed_label_matches"] = [
                m.to_api_dict() for m in allowed_label_matches
            ]
        if allowed_label_matches_id is not None:
            payload["allowed_label_matches_id"] = allowed_label_matches_id
        if exclusion_rules is not None:
            payload[EXCLUSION_RULES_CAMEL_KEY] = [
                rule.to_api_dict() if hasattr(rule, "to_api_dict") else rule
                for rule in exclusion_rules
            ]
        result = self.post(payload, f"benchmarks/{benchmark_id}/evaluationsV2")
        eval_id = result.get(EVALUATION_ID_KEY)
        if not eval_id:
            raise RuntimeError(
                f"Unexpected create benchmark evaluation V2 response: {result}"
            )
        return self.get_evaluation_v2(str(eval_id))

    def leaderboard_ranking(
        self,
        metric_type: str,
        benchmark_ids: List[str],
        *,
        confidence_threshold: Optional[float] = None,
        model_ids: Optional[List[str]] = None,
        scope: Optional[str] = None,
        collapse: Optional[str] = None,
    ) -> List[LeaderboardRankingEntry]:
        """Rank model runs on one or more benchmarks by a metric.

        Parameters:
            metric_type: Metric to rank by — one of ``"MAP_50"``,
                ``"MAP_50_95"``, ``"AP_SMALL"``, ``"AP_MEDIUM"``,
                ``"AP_LARGE"``, ``"PRECISION"``, ``"RECALL"``, ``"F1"``.
            benchmark_ids: Benchmark ids (``bm_*``) to rank across.
            confidence_threshold: Confidence operating point for
                ``PRECISION`` / ``RECALL`` / ``F1``.
            model_ids: Optional model ids to restrict the ranking to.
            scope: ``"mine"`` (only the caller's evaluations) or ``"all"``
                (default).
            collapse: ``"bestPerModel"`` (default), ``"allRuns"``, or
                ``"allEvaluations"``.

        Returns:
            List of :class:`LeaderboardRankingEntry`, best score first.
        """
        payload: Dict[str, Any] = {
            METRIC_TYPE_KEY: metric_type,
            BENCHMARK_IDS_KEY: benchmark_ids,
        }
        if confidence_threshold is not None:
            payload[CONFIDENCE_THRESHOLD_KEY] = confidence_threshold
        if model_ids is not None:
            payload[MODEL_IDS_KEY] = model_ids
        if scope is not None:
            payload[SCOPE_KEY] = scope
        if collapse is not None:
            payload[COLLAPSE_KEY] = collapse
        rows = self.post(payload, "leaderboard/ranking")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected leaderboard ranking response: {rows!r}"
            )
        return [LeaderboardRankingEntry.parse_obj(r) for r in rows]

    def leaderboard_f1_curve(
        self,
        benchmark_ids: List[str],
        *,
        model_ids: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> List[LeaderboardF1CurveEntry]:
        """Return F1-vs-confidence curves for the top runs on benchmarks.

        Parameters:
            benchmark_ids: Benchmark ids (``bm_*``).
            model_ids: Optional model ids to restrict the curves to.
            top_n: Number of top-ranked runs to return curves for (default 5).

        Returns:
            List of :class:`LeaderboardF1CurveEntry`, best F1 first.
        """
        payload: Dict[str, Any] = {
            BENCHMARK_IDS_KEY: benchmark_ids,
            TOP_N_KEY: top_n,
        }
        if model_ids is not None:
            payload[MODEL_IDS_KEY] = model_ids
        rows = self.post(payload, "leaderboard/f1Curve")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"Unexpected leaderboard F1 curve response: {rows!r}"
            )
        return [LeaderboardF1CurveEntry.parse_obj(r) for r in rows]

    def get_evaluation_v2_filter_schema(
        self, evaluation_id: str
    ) -> EvaluationV2FilterSchema:
        """Return the filter vocabulary for an evaluation.

        Parameters:
            evaluation_id: Evaluation id (``evalv2_*``).

        Returns:
            :class:`~nucleus.data_transfer_object.evaluation_v2.EvaluationV2FilterSchema`.
        """
        data = self.get(f"evaluationsV2/{evaluation_id}/filterSchema")
        return EvaluationV2FilterSchema.parse_obj(data)

    @deprecated(msg="Prefer calling Dataset.info() directly.")
    def dataset_info(self, dataset_id: str):
        dataset = self.get_dataset(dataset_id)
        return dataset.info()

    @deprecated(
        "Model runs have been deprecated and will be removed. Use a Model instead."
    )
    def model_run_info(self, model_run_id: str):
        # TODO: deprecate ModelRun
        return self.make_request(
            {}, f"modelRun/{model_run_id}/info", requests.get
        )

    @deprecated("Prefer calling Dataset.refloc instead.")
    @sanitize_string_args
    def dataitem_ref_id(self, dataset_id: str, reference_id: str):
        # TODO: deprecate in favor of Dataset.refloc invocation
        return self.make_request(
            {}, f"dataset/{dataset_id}/refloc/{reference_id}", requests.get
        )

    @deprecated("Prefer calling Dataset.predictions_refloc instead.")
    @sanitize_string_args
    def predictions_ref_id(
        self, model_run_id: str, ref_id: str, dataset_id: Optional[str] = None
    ):
        if dataset_id:
            raise RuntimeError(
                "Need to pass a dataset id. Or use Dataset.predictions_refloc."
            )
        # TODO: deprecate ModelRun
        m_run = self.get_model_run(model_run_id, dataset_id)
        return m_run.refloc(ref_id)

    @deprecated("Prefer calling Dataset.iloc instead.")
    def dataitem_iloc(self, dataset_id: str, i: int):
        # TODO: deprecate in favor of Dataset.iloc invocation
        return self.make_request(
            {}, f"dataset/{dataset_id}/iloc/{i}", requests.get
        )

    @deprecated("Prefer calling Dataset.predictions_iloc instead.")
    def predictions_iloc(self, model_run_id: str, i: int):
        # TODO: deprecate ModelRun
        return self.make_request(
            {}, f"modelRun/{model_run_id}/iloc/{i}", requests.get
        )

    @deprecated("Prefer calling Dataset.loc instead.")
    def dataitem_loc(self, dataset_id: str, dataset_item_id: str):
        # TODO: deprecate in favor of Dataset.loc invocation
        return self.make_request(
            {}, f"dataset/{dataset_id}/loc/{dataset_item_id}", requests.get
        )

    @deprecated("Prefer calling Dataset.predictions_loc instead.")
    def predictions_loc(self, model_run_id: str, dataset_item_id: str):
        # TODO: deprecate ModelRun
        return self.make_request(
            {}, f"modelRun/{model_run_id}/loc/{dataset_item_id}", requests.get
        )

    @deprecated("Prefer calling Dataset.create_slice instead.")
    def create_slice(self, dataset_id: str, payload: dict) -> Slice:
        # TODO: deprecate in favor of Dataset.create_slice
        dataset = self.get_dataset(dataset_id)
        return dataset.create_slice(payload["name"], payload["reference_ids"])

    def get_slice(self, slice_id: str) -> Slice:
        # TODO: migrate to Dataset method and deprecate
        """Returns a slice object by Nucleus-generated ID.

        Parameters:
            slice_id: Nucleus-generated slice ID (starts with ``slc_``). This can
              be retrieved via :meth:`Dataset.slices` or a Nucleus dashboard URL.

        Returns:
            :class:`Slice`: The Nucleus slice as an object.
        """
        return Slice(slice_id, self)

    @deprecated("Prefer calling Slice.info instead.")
    def slice_info(self, slice_id: str) -> dict:
        # TODO: deprecate in favor of Slice.info
        response = self.make_request(
            {},
            f"slice/{slice_id}",
            requests_command=requests.get,
        )
        return response

    def delete_slice(self, slice_id: str) -> dict:
        # TODO: migrate to Dataset method and deprecate
        """Deletes slice from Nucleus.

        Parameters:
            slice_id: Nucleus-generated slice ID (starts with ``slc_``). This can
              be retrieved via :meth:`Dataset.slices` or a Nucleus dashboard URL.

        Returns:
            Empty payload response.
        """
        response = self.make_request(
            {},
            f"slice/{slice_id}",
            requests_command=requests.delete,
        )
        return response

    @deprecated("Prefer calling Dataset.delete_annotations instead.")
    def delete_annotations(
        self,
        dataset_id: str,
        reference_ids: Optional[list] = None,
        keep_history=True,
    ) -> AsyncJob:
        dataset = self.get_dataset(dataset_id)
        return dataset.delete_annotations(reference_ids, keep_history)

    def append_to_slice(
        self,
        slice_id: str,
        reference_ids: List[str],
        dataset_id: str,
    ) -> dict:
        # TODO: migrate to Slice method and deprecate
        """Appends dataset items or scenes to an existing slice.

        Parameters:
            slice_id: Nucleus-generated dataset ID (starts with ``slc_``). This can
              be retrieved via :meth:`Dataset.slices` or a Nucleus dashboard URL.
            reference_ids: List of user-defined reference IDs of dataset items or scenes
              to append to the slice.
            dataset_id: ID of dataset this slice belongs to.

        Returns:
            Empty payload response.
        """

        response = self.make_request(
            {REFERENCE_IDS_KEY: reference_ids, DATASET_ID_KEY: dataset_id},
            f"slice/{slice_id}/append",
        )
        return response

    def list_autotags(self, dataset_id: str) -> List[dict]:
        # TODO: deprecate in favor of Dataset.list_autotags invocation
        response = self.make_request(
            {},
            f"{dataset_id}/list_autotags",
            requests_command=requests.get,
        )
        if isinstance(response, dict) and AUTOTAGS_KEY in response:
            return list(response[AUTOTAGS_KEY])
        if isinstance(response, list):
            return list(response)
        return []

    def delete_autotag(self, autotag_id: str) -> dict:
        # TODO: migrate to Dataset method (use autotag name, not id) and deprecate
        """Deletes an autotag by ID.

        Parameters:
            autotag_id: Nucleus-generated autotag ID (starts with ``tag_``). This can
              be retrieved via :meth:`list_autotags` or a Nucleus dashboard URL.

        Returns:
            Empty payload response.
        """
        return self.make_request({}, f"autotag/{autotag_id}", requests.delete)

    def get_autotag_refinement_metrics(self, autotag_id: str) -> dict:
        """Retrieves refinement metrics for an autotag by ID.

        Parameters:
            autotag_id: Nucleus-generated autotag ID (starts with ``tag_``). This can
              be retrieved via :meth:`list_autotags` or a Nucleus dashboard URL.

        Returns:
            Response payload::

                {
                    "total_refinement_steps": int
                    "average_positives_selected_per_refinement": int
                    "average_ms_taken_in_refinement": float
                }
        """
        return self.make_request(
            {}, f"autotag/{autotag_id}/refinementMetrics", requests.get
        )

    def delete_model(self, model_id: str) -> dict:
        """Deletes a model by ID.

        Parameters:
            model_id: Nucleus-generated model ID (starts with ``prj_``). This can
              be retrieved via :meth:`list_models` or a Nucleus dashboard URL.

        Returns:
            Empty payload response.
        """
        response = self.make_request(
            {},
            f"model/{model_id}",
            requests_command=requests.delete,
        )
        return response

    def upload_model_weights(
        self,
        model: Union[Model, str],
        path: str,
        *,
        content_type: Optional[str] = None,
        original_filename: Optional[str] = None,
        checksum_sha256: Optional[str] = None,
        progress: bool = True,
    ) -> ModelWeights:
        """Attach a weights artifact to a model.

        Any binary is accepted — there are no format constraints — up to 10 GB.
        Requires edit access on the model.

        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            model = client.get_model(reference_id="My-CNN")
            client.upload_model_weights(model, "/path/to/weights.bin")

        Parameters:
            model: A :class:`Model` or a model id (``prj_*``).
            path: Local path of the artifact to upload.
            content_type: Content type to record for the artifact. Defaults to
              ``application/octet-stream``.
            original_filename: Filename to show for the artifact. Defaults to
              the name of the file at ``path``.
            checksum_sha256: Optional SHA-256 of the artifact.
            progress: Whether to show a ``tqdm`` progress bar for the upload.

        Returns:
            :class:`ModelWeights`: Metadata for the uploaded artifact.
        """
        model_id = model.id if isinstance(model, Model) else model
        path = os.path.expanduser(path)
        filename = (
            original_filename
            if original_filename is not None
            else os.path.basename(path)
        )
        total_bytes = os.path.getsize(path)
        if total_bytes > MODEL_WEIGHTS_MAX_BYTES:
            raise ValueError(
                f"{path} is {total_bytes} bytes, which exceeds the "
                f"{MODEL_WEIGHTS_MAX_BYTES // 1024 ** 3} GB model weights limit"
            )

        presign = self.make_request(
            _presign_payload(
                total_bytes, content_type, filename, checksum_sha256
            ),
            f"model/{model_id}/weights/presign",
        )
        upload_id = presign.get(UPLOAD_ID_KEY)
        if not upload_id:
            raise ValueError(
                "Presign response did not include an uploadId; cannot upload"
            )
        progress_bar = (
            self.tqdm_bar(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=f"Uploading {filename}",
            )
            if progress
            else None
        )
        try:
            on_progress = (
                _progress_to_bar(progress_bar)
                if progress_bar is not None
                else None
            )
            parts = _transfer_weights_to_storage(
                path, presign, total_bytes, on_progress
            )
            finalized = self.make_request(
                _finalize_payload(upload_id, parts),
                f"model/{model_id}/weights/finalize",
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return ModelWeights.from_json(finalized)

    def download_model_weights(
        self,
        model: Union[Model, str],
        path: str,
        *,
        progress: bool = True,
    ) -> str:
        """Download a model's weights artifact to a local path.

        Available to anyone who can see the model.

        ::

            import nucleus

            client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
            model = client.get_model(reference_id="My-CNN")
            client.download_model_weights(model, "/path/to/save/weights.bin")

        Parameters:
            model: A :class:`Model` or a model id (``prj_*``).
            path: Local path to write the artifact to. Parent directories are
              created if needed.
            progress: Whether to show a ``tqdm`` progress bar for the download.

        Returns:
            str: The path written.

        Raises:
            NotFoundError: If the model has no weights artifact to download.
        """
        model_id = model.id if isinstance(model, Model) else model
        path = os.path.expanduser(path)
        # Ask for the URL as JSON rather than following the redirect, so the
        # API credentials aren't replayed to the download host.
        signed = self.make_request(
            {},
            f"model/{model_id}/weights/download?json=1",
            requests_command=requests.get,
        )
        url = signed.get(URL_KEY)
        if not url:
            raise NotFoundError(
                f"Model {model_id} has no downloadable weights artifact"
            )
        if not progress:
            return _stream_weights_to_file(url, path)
        # The size isn't known until the GET responds, so the bar tracks bytes
        # without a percentage.
        progress_bar = self.tqdm_bar(
            unit="B",
            unit_scale=True,
            desc=f"Downloading {os.path.basename(path)}",
        )
        try:
            return _stream_weights_to_file(
                url, path, _progress_to_bar(progress_bar)
            )
        finally:
            progress_bar.close()

    def get_model_weights(self, model: Union[Model, str]) -> ModelWeights:
        """Fetch metadata for a model's weights artifact.

        Parameters:
            model: A :class:`Model` or a model id (``prj_*``).

        Returns:
            :class:`ModelWeights`: Metadata. ``present`` is ``False`` when the
            model has no weights artifact available.
        """
        model_id = model.id if isinstance(model, Model) else model
        return ModelWeights.from_json(
            self.make_request(
                {}, f"model/{model_id}/weights", requests_command=requests.get
            )
        )

    def delete_model_weights(self, model: Union[Model, str]) -> bool:
        """Delete a model's weights artifact.

        Requires edit access on the model.

        Parameters:
            model: A :class:`Model` or a model id (``prj_*``).

        Returns:
            bool: Whether an artifact was deleted.
        """
        model_id = model.id if isinstance(model, Model) else model
        response = self.make_request(
            {},
            f"model/{model_id}/weights",
            requests_command=requests.delete,
        )
        return bool(response.get(DELETED_KEY, False))

    def download_pointcloud_task(
        self, task_id: str, frame_num: int
    ) -> List[Union[Point3D, LidarPoint]]:
        """
        Download the lidar point cloud data for a given task and frame number.

        Parameters:
            task_id: download point cloud for this particular task
            frame_num: download point cloud for this particular frame

        Returns:
            List of Point3D objects

        """

        response = self.make_request(
            payload={},
            route=f"task/{task_id}/frame/{frame_num}",
            requests_command=requests.get,
        )
        points = response.get(POINTS_KEY, None)
        if points is None or len(points) == 0:
            raise RuntimeError("Response has invalid payload")

        sample_point = points[0]
        if I_KEY in sample_point.keys():
            return [LidarPoint.from_json(pt) for pt in points]

        return [Point3D.from_json(pt) for pt in points]

    def download_pointcloud_tasks(
        self, task_ids: List[str], frame_num: int
    ) -> Dict[str, List[Union[Point3D, LidarPoint]]]:
        """
        Download the lidar point cloud data for a given set of tasks and frame number.

        Parameters:
            task_ids: list of task ids to fetch data from
            frame_num: download point cloud for this particular frame

        Returns:
            A dictionary from task_id to list of Point3D objects

        """
        endpoints = [
            f"task/{task_id}/frame/{frame_num}" for task_id in task_ids
        ]
        progressbar = self.tqdm_bar(
            total=len(endpoints),
            desc="Downloading pointcloud tasks",
        )
        results = make_multiple_requests_concurrently(
            client=self,
            requests=endpoints,
            route=None,
            progressbar=progressbar,
        )
        resp = {}

        for result in results:
            req, data = result
            task_id = req.split("/")[1]  # task/<task id>/frame/1 => task_id
            points = data.get(POINTS_KEY, None)
            if points is None or len(points) == 0:
                raise RuntimeError("Response has invalid payload")

            sample_point = points[0]
            if I_KEY in sample_point.keys():
                resp[task_id] = [LidarPoint.from_json(pt) for pt in points]
            else:
                resp[task_id] = [Point3D.from_json(pt) for pt in points]

        return resp

    @deprecated("Prefer calling Dataset.create_custom_index instead.")
    def create_custom_index(
        self, dataset_id: str, embeddings_urls: list, embedding_dim: int
    ):
        # TODO: deprecate in favor of Dataset.create_custom_index invocation
        dataset = self.get_dataset(dataset_id)
        return dataset.create_custom_index(
            embeddings_urls=embeddings_urls, embedding_dim=embedding_dim
        )

    @deprecated("Prefer calling Dataset.delete_custom_index instead.")
    def delete_custom_index(self, dataset_id: str, image: bool):
        # TODO: deprecate in favor of Dataset.delete_custom_index invocation
        return self.make_request(
            {"image": image},
            f"indexing/{dataset_id}",
            requests_command=requests.delete,
        )

    @deprecated("Prefer calling Dataset.set_primary_index instead.")
    def set_primary_index(self, dataset_id: str, image: bool, custom: bool):
        # TODO: deprecate in favor of Dataset.set_primary_index invocation
        return self.make_request(
            {"image": image, "custom": custom},
            f"indexing/{dataset_id}/setPrimary",
            requests_command=requests.post,
        )

    @deprecated("Prefer calling Dataset.set_continuous_indexing instead.")
    def set_continuous_indexing(self, dataset_id: str, enable: bool = True):
        # TODO: deprecate in favor of Dataset.set_continuous_indexing invocation
        return self.make_request(
            {INDEX_CONTINUOUS_ENABLE_KEY: enable},
            f"indexing/{dataset_id}/setContinuous",
            requests_command=requests.post,
        )

    @deprecated("Prefer calling Dataset.create_image_index instead.")
    def create_image_index(self, dataset_id: str):
        # TODO: deprecate in favor of Dataset.create_image_index invocation
        return self.make_request(
            {},
            f"indexing/{dataset_id}/internal/image",
            requests_command=requests.post,
        )

    @deprecated("Prefer calling Dataset.create_object_index instead.")
    def create_object_index(
        self, dataset_id: str, model_run_id: str, gt_only: bool
    ):
        # TODO: deprecate in favor of Dataset.create_object_index invocation
        payload: Dict[str, Union[str, bool]] = {}
        if model_run_id:
            payload["model_run_id"] = model_run_id
        elif gt_only:
            payload["ingest_gt_only"] = True
        return self.make_request(
            payload,
            f"indexing/{dataset_id}/internal/object",
            requests_command=requests.post,
        )

    def delete(self, route: str):
        return self.connection.delete(route)

    def get(self, route: str):
        return self.connection.get(route)

    def patch(self, payload: dict, route: str):
        return self.connection.patch(payload, route)

    def post(self, payload: dict, route: str):
        return self.connection.post(payload, route)

    def put(self, payload: dict, route: str):
        return self.connection.put(payload, route)

    # TODO: Fix return type, can be a list as well. Brings on a lot of mypy errors ...
    def make_request(
        self,
        payload: Optional[dict],
        route: str,
        requests_command=requests.post,
        return_raw_response: bool = False,
    ) -> Union[dict, Any]:
        """Makes a request to a Nucleus API endpoint.

        Logs a warning if not successful.

        Parameters:
            payload: Given request payload.
            route: Route for the request.
            requests_command: ``requests.post``, ``requests.get``, or ``requests.delete``.
            return_raw_response: Whether to return the raw response object.

        Returns:
            Response payload as JSON dict or request object.
        """
        if payload is None:
            payload = {}
        if requests_command is requests.get:
            if payload:
                print(
                    "Received defined payload with GET request! Will ignore payload"
                )
            payload = None
        return self.connection.make_request(payload, route, requests_command, return_raw_response)  # type: ignore

    def _set_api_key(self, api_key):
        """Fetch API key from environment variable NUCLEUS_API_KEY if not set"""
        api_key = api_key if api_key else os.environ.get("NUCLEUS_API_KEY")
        if api_key is None:
            raise NoAPIKey()

        return api_key

    @staticmethod
    def valid_dirname(dirname) -> str:
        """Validates that a directory exists.

        Parameters:
            dirname: Path of directory.

        Returns:
            Existing directory path.
        """
        # ensures path ends with a slash
        _dirname = os.path.join(os.path.expanduser(dirname), "")
        if not os.path.exists(_dirname):
            raise ValueError(
                f"Given directory name: {dirname} does not exists. Searched in {_dirname}"
            )
        return _dirname

    def create_dataset_from_dir(
        self,
        dirname: str,
        dataset_name: Optional[str] = None,
        use_privacy_mode: bool = False,
        privacy_mode_proxy: str = "",
        allowed_file_types: Tuple[str, ...] = ("png", "jpg", "jpeg"),
        skip_size_warning: bool = False,
    ) -> Dataset:
        """
        Create a dataset by recursively crawling through a directory.
        A DatasetItem will be created for each unique image found.

        Parameters:
            dirname: Where to look for image files, recursively
            dataset_name: If none is given, the parent folder name is used
            use_privacy_mode: Whether the dataset should be treated as privacy
            privacy_mode_proxy: Endpoint that serves image files for privacy mode, ignore if not using privacy mode.
                The proxy should work based on the relative path of the images in the directory.
            allowed_file_types: Which file type extensions to search for, ie: ('jpg', 'png')
            skip_size_warning: If False, it will throw an error if the script globs more than 500 images. This is a safety check in case the dirname has a typo, and grabs too much data.
        """
        existing_dirname = self.valid_dirname(dirname)
        folder_name = os.path.basename(existing_dirname.rstrip("/"))
        dataset_name = dataset_name or folder_name
        dataset = self.create_dataset(
            name=dataset_name, use_privacy_mode=use_privacy_mode
        )
        job = dataset.add_items_from_dir(
            existing_dirname=existing_dirname,
            privacy_mode_proxy=privacy_mode_proxy,
            allowed_file_types=allowed_file_types,
            skip_size_warning=skip_size_warning,
        )
        if job is not None:
            job.sleep_until_complete()
        return dataset
