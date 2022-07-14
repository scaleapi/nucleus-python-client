"""Nucleus Python SDK. """

__all__ = [
    "AsyncJob",
    "BoxAnnotation",
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
    "Frame",
    "Keypoint",
    "KeypointsAnnotation",
    "KeypointsPrediction",
    "LidarScene",
    "LineAnnotation",
    "LinePrediction",
    "Model",
    "ModelCreationError",
    # "MultiCategoryAnnotation", # coming soon!
    "NotFoundError",
    "NucleusAPIError",
    "NucleusClient",
    "Point",
    "Point3D",
    "PolygonAnnotation",
    "PolygonPrediction",
    "Quaternion",
    "Segment",
    "SegmentationAnnotation",
    "SegmentationPrediction",
    "Slice",
    "VideoScene",
]

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import pkg_resources
import pydantic
import requests
import tqdm
import tqdm.notebook as tqdm_notebook

from nucleus.url_utils import sanitize_string_args

from .annotation import (
    BoxAnnotation,
    CategoryAnnotation,
    CuboidAnnotation,
    Keypoint,
    KeypointsAnnotation,
    LineAnnotation,
    MultiCategoryAnnotation,
    Point,
    Point3D,
    PolygonAnnotation,
    Segment,
    SegmentationAnnotation,
)
from .camera_params import CameraParams
from .connection import Connection
from .constants import (
    ANNOTATION_METADATA_SCHEMA_KEY,
    ANNOTATIONS_IGNORED_KEY,
    ANNOTATIONS_PROCESSED_KEY,
    AUTOTAGS_KEY,
    DATASET_ID_KEY,
    DATASET_IS_SCENE_KEY,
    DEFAULT_NETWORK_TIMEOUT_SEC,
    EMBEDDING_DIMENSION_KEY,
    EMBEDDINGS_URL_KEY,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    ERRORS_KEY,
    IMAGE_KEY,
    IMAGE_URL_KEY,
    INDEX_CONTINUOUS_ENABLE_KEY,
    ITEM_METADATA_SCHEMA_KEY,
    ITEMS_KEY,
    JOB_CREATION_TIME_KEY,
    JOB_ID_KEY,
    JOB_LAST_KNOWN_STATUS_KEY,
    JOB_TYPE_KEY,
    KEEP_HISTORY_KEY,
    MESSAGE_KEY,
    MODEL_RUN_ID_KEY,
    MODEL_TAGS_KEY,
    NAME_KEY,
    NUCLEUS_ENDPOINT,
    PREDICTIONS_IGNORED_KEY,
    PREDICTIONS_PROCESSED_KEY,
    REFERENCE_IDS_KEY,
    SLICE_ID_KEY,
    STATUS_CODE_KEY,
    UPDATE_KEY,
)
from .data_transfer_object.dataset_details import DatasetDetails
from .data_transfer_object.dataset_info import DatasetInfo
from .dataset import Dataset
from .dataset_item import DatasetItem
from .deprecation_warning import deprecated
from .errors import (
    DatasetItemRetrievalError,
    ModelCreationError,
    ModelRunCreationError,
    NoAPIKey,
    NotFoundError,
    NucleusAPIError,
)
from .job import AsyncJob
from .logger import logger
from .model import Model
from .model_run import ModelRun
from .payload_constructor import (
    construct_annotation_payload,
    construct_append_payload,
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
    SegmentationPrediction,
)
from .quaternion import Quaternion
from .retry_strategy import RetryStrategy
from .scene import Frame, LidarScene, VideoScene
from .slice import Slice
from .upload_response import UploadResponse
from .validate import Validate

# pylint: disable=E1101
# TODO: refactor to reduce this file to under 1000 lines.
# pylint: disable=C0302


__version__ = pkg_resources.get_distribution("scale-nucleus").version


class NucleusClient:
    """Client to interact with the Nucleus API via Python SDK.

    Parameters:
        api_key: Follow `this guide <https://scale.com/docs/account#section-api-keys>`_
          to retrieve your API keys.
        use_notebook: Whether the client is being used in a notebook (toggles tqdm
          style). Default is ``False``.
        endpoint: Base URL of the API. Default is Nucleus's current production API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_notebook: bool = False,
        endpoint: str = None,
    ):
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
            self.tqdm_bar = tqdm_notebook.tqdm
        self._connection = Connection(self.api_key, self.endpoint)
        self.validate = Validate(self.api_key, self.endpoint)

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
        dataset_details = pydantic.parse_obj_as(List[DatasetDetails], response)
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
            )
            for model in model_objects["models"]
        ]

    @property
    def jobs(
        self,
    ) -> List[AsyncJob]:
        """Lists all jobs, see NucleusClinet.list_jobs(...) for advanced options

        Returns:
            List of all AsyncJobs
        """
        return self.list_jobs()

    @deprecated(msg="Use the NucleusClient.models property in the future.")
    def list_models(self) -> List[Model]:
        return self.models

    @deprecated(msg="Use the NucleusClient.datasets property in the future.")
    def list_datasets(self) -> Dict[str, Union[str, List[str]]]:
        return self.make_request({}, "dataset/", requests.get)

    def list_jobs(
        self, show_completed=None, date_limit=None
    ) -> List[AsyncJob]:
        """Fetches all of your running jobs in Nucleus.

        Parameters:
            show_completed: Whether to fetch completed and errored jobs or just
              running jobs. Default behavior is False.
            date_limit: Only fetch jobs that were started after this date. Default
              behavior is 2 weeks prior to the current date.

        Returns:
            List[:class:`AsyncJob`]: List of running asynchronous jobs
            associated with the client API key.
        """
        # TODO: What type is date_limit? Use pydantic ...
        payload = {show_completed: show_completed, date_limit: date_limit}
        job_objects = self.make_request(payload, "jobs/", requests.get)
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
        """Fetches a dataset by its ID.

        Parameters:
            job_id: The ID of the dataset to fetch.

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
        self, model_id: str = None, model_run_id: str = None
    ) -> Model:
        """Fetches a model by its ID.

        Parameters:
            model_id: You can pass either a model ID (starts with ``prj_``) or a model run ID (starts with ``run_``) This can be retrieved via :meth:`list_models` or a Nucleus dashboard URL. Model run IDs result from the application of a model to a dataset.
            model_run_id: You can pass either a model ID (starts with ``prj_``), or a model run ID (starts with ``run_``) This can
              be retrieved via :meth:`list_models` or a Nucleus dashboard URL. Model run IDs result from the application of a model to a dataset.

              In the future, we plan to hide ``model_run_ids`` fully from users.

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

    def create_dataset_from_project(
        self, project_id: str, last_n_tasks: int = None, name: str = None
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
            item_metadata_schema: Dict defining item-level metadata schema. See below.
            annotation_metadata_schema: Dict defining annotation-level metadata schema.

                Metadata schemas must be structured as follows::

                    {
                        "field_name": {
                            "type": "category" | "number" | "text"
                            "choices": List[str] | None
                            "description": str | None
                        },
                        ...
                    }

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
                ANNOTATION_METADATA_SCHEMA_KEY: annotation_metadata_schema,
                ITEM_METADATA_SCHEMA_KEY: item_metadata_schema,
            },
            "dataset/create",
        )
        return Dataset(response[DATASET_ID_KEY], self)

    def delete_dataset(self, dataset_id: str) -> dict:
        """
        Deletes a dataset by ID.

        All items, annotations, and predictions associated with the dataset will
        be deleted as well.

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

    @deprecated("Use Dataset.append instead.")
    def populate_dataset(
        self,
        dataset_id: str,
        dataset_items: List[DatasetItem],
        batch_size: int = 20,
        update: bool = False,
    ):
        dataset = self.get_dataset(dataset_id)
        return dataset.append(
            dataset_items, batch_size=batch_size, update=update
        )

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
                name, reference_id, metadata, bundle_name, tags
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
        )

    def create_launch_model(
        self,
        name: str,
        reference_id: str,
        bundle_args: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ) -> Model:
        """
        Adds a :class:`Model` to Nucleus, as well as a Launch bundle from a given function.

        Parameters:
            name: A human-readable name for the model.
            reference_id: Unique, user-controlled ID for the model. This can be
              used, for example, to link to an external storage of models which
              may have its own id scheme.
            bundle_args: Dict for kwargs for the creation of a Launch bundle,
              more details on the keys below.
            metadata: An arbitrary dictionary of additional data about this model
              that can be stored and retrieved. For example, you can store information
              about the hyperparameters used in training this model.

        Returns:
            :class:`Model`: The newly created model as an object.

        Details on `bundle_args`:
                Grabs a s3 signed url and uploads a model bundle to Scale Launch.

        A model bundle consists of exactly {predict_fn_or_cls}, {load_predict_fn + model}, or {load_predict_fn + load_model_fn}.
        Pre/post-processing code can be included inside load_predict_fn/model or in predict_fn_or_cls call.

        Parameters:
            model_bundle_name: Name of model bundle you want to create. This acts as a unique identifier.
            predict_fn_or_cls: Function or a Callable class that runs end-to-end (pre/post processing and model inference) on the call.
                I.e. `predict_fn_or_cls(REQUEST) -> RESPONSE`.
            model: Typically a trained Neural Network, e.g. a Pytorch module
            load_predict_fn: Function that when called with model, returns a function that carries out inference
                I.e. `load_predict_fn(model) -> func; func(REQUEST) -> RESPONSE`
            load_model_fn: Function that when run, loads a model, e.g. a Pytorch module
                I.e. `load_predict_fn(load_model_fn()) -> func; func(REQUEST) -> RESPONSE`
            bundle_url: Only for self-hosted mode. Desired location of bundle.
            Overrides any value given by self.bundle_location_fn
            requirements: A list of python package requirements, e.g.
                ["tensorflow==2.3.0", "tensorflow-hub==0.11.0"]. If no list has been passed, will default to the currently
                imported list of packages.
            app_config: Either a Dictionary that represents a YAML file contents or a local path to a YAML file.
            env_params: A dictionary that dictates environment information e.g.
                the use of pytorch or tensorflow, which cuda/cudnn versions to use.
                Specifically, the dictionary should contain the following keys:
                "framework_type": either "tensorflow" or "pytorch".
                "pytorch_version": Version of pytorch, e.g. "1.5.1", "1.7.0", etc. Only applicable if framework_type is pytorch
                "cuda_version": Version of cuda used, e.g. "11.0".
                "cudnn_version" Version of cudnn used, e.g. "cudnn8-devel".
                "tensorflow_version": Version of tensorflow, e.g. "2.3.0". Only applicable if framework_type is tensorflow
            globals_copy: Dictionary of the global symbol table. Normally provided by `globals()` built-in function.
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

        bundle = launch_client.create_model_bundle(**kwargs)
        return self.create_model(
            name,
            reference_id,
            metadata,
            bundle.name,
        )

    def create_launch_model_from_dir(
        self,
        name: str,
        reference_id: str,
        bundle_from_dir_args: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ) -> Model:
        """
        Adds a :class:`Model` to Nucleus, as well as a Launch bundle from a directory.

        Parameters:
            name: A human-readable name for the model.
            reference_id: Unique, user-controlled ID for the model. This can be
              used, for example, to link to an external storage of models which
              may have its own id scheme.
            bundle_from_dir_args: Dict for kwargs for the creation of a bundle from directory,
              more details on the keys below.
            metadata: An arbitrary dictionary of additional data about this model
              that can be stored and retrieved. For example, you can store information
              about the hyperparameters used in training this model.

        Returns:
            :class:`Model`: The newly created model as an object.

        Details on `bundle_from_dir_args`
        Packages up code from one or more local filesystem folders and uploads them as a bundle to Scale Launch.
        In this mode, a bundle is just local code instead of a serialized object.

        For example, if you have a directory structure like so, and your current working directory is also `my_root`:

        ```
        my_root/
            my_module1/
                __init__.py
                ...files and directories
                my_inference_file.py
            my_module2/
                __init__.py
                ...files and directories
        ```

        then calling `create_model_bundle_from_dirs` with `base_paths=["my_module1", "my_module2"]` essentially
        creates a zip file without the root directory, e.g.:

        ```
        my_module1/
            __init__.py
            ...files and directories
            my_inference_file.py
        my_module2/
            __init__.py
            ...files and directories
        ```

        and these contents will be unzipped relative to the server side `PYTHONPATH`. Bear these points in mind when
        referencing Python module paths for this bundle. For instance, if `my_inference_file.py` has `def f(...)`
        as the desired inference loading function, then the `load_predict_fn_module_path` argument should be
        `my_module1.my_inference_file.f`.


        Keys for `bundle_from_dir_args`:
            model_bundle_name: Name of model bundle you want to create. This acts as a unique identifier.
            base_paths: The paths on the local filesystem where the bundle code lives.
            requirements_path: A path on the local filesystem where a requirements.txt file lives.
            env_params: A dictionary that dictates environment information e.g.
                the use of pytorch or tensorflow, which cuda/cudnn versions to use.
                Specifically, the dictionary should contain the following keys:
                "framework_type": either "tensorflow" or "pytorch".
                "pytorch_version": Version of pytorch, e.g. "1.5.1", "1.7.0", etc. Only applicable if framework_type is pytorch
                "cuda_version": Version of cuda used, e.g. "11.0".
                "cudnn_version" Version of cudnn used, e.g. "cudnn8-devel".
                "tensorflow_version": Version of tensorflow, e.g. "2.3.0". Only applicable if framework_type is tensorflow
            load_predict_fn_module_path: A python module path for a function that, when called with the output of
                load_model_fn_module_path, returns a function that carries out inference.
            load_model_fn_module_path: A python module path for a function that returns a model. The output feeds into
                the function located at load_predict_fn_module_path.
            app_config: Either a Dictionary that represents a YAML file contents or a local path to a YAML file.
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

        bundle = launch_client.create_model_bundle_from_dir(**kwargs)

        return self.create_model(
            name,
            reference_id,
            metadata,
            bundle.name,
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
            slice_id: Nucleus-generated dataset ID (starts with ``slc_``). This can
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
            slice_id: Nucleus-generated dataset ID (starts with ``slc_``). This can
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
        self, dataset_id: str, reference_ids: list = None, keep_history=True
    ) -> AsyncJob:
        dataset = self.get_dataset(dataset_id)
        return dataset.delete_annotations(reference_ids, keep_history)

    def append_to_slice(
        self,
        slice_id: str,
        reference_ids: List[str],
    ) -> dict:
        # TODO: migrate to Slice method and deprecate
        """Appends dataset items to an existing slice.

        Parameters:
            slice_id: Nucleus-generated dataset ID (starts with ``slc_``). This can
              be retrieved via :meth:`Dataset.slices` or a Nucleus dashboard URL.
            reference_ids: List of user-defined reference IDs of the dataset items
              to append to the slice.

        Returns:
            Empty payload response.
        """

        response = self.make_request(
            {REFERENCE_IDS_KEY: reference_ids}, f"slice/{slice_id}/append"
        )
        return response

    def list_autotags(self, dataset_id: str) -> List[dict]:
        # TODO: deprecate in favor of Dataset.list_autotags invocation
        response = self.make_request(
            {},
            f"{dataset_id}/list_autotags",
            requests_command=requests.get,
        )
        return response[AUTOTAGS_KEY] if AUTOTAGS_KEY in response else response

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
        return self._connection.delete(route)

    def get(self, route: str):
        return self._connection.get(route)

    def post(self, payload: dict, route: str):
        return self._connection.post(payload, route)

    def put(self, payload: dict, route: str):
        return self._connection.put(payload, route)

    # TODO: Fix return type, can be a list as well. Brings on a lot of mypy errors ...
    def make_request(
        self,
        payload: Optional[dict],
        route: str,
        requests_command=requests.post,
    ) -> dict:
        """Makes a request to a Nucleus API endpoint.

        Logs a warning if not successful.

        Parameters:
            payload: Given request payload.
            route: Route for the request.
            Requests command: ``requests.post``, ``requests.get``, or ``requests.delete``.

        Returns:
            Response payload as JSON dict.
        """
        if payload is None:
            payload = {}
        if requests_command is requests.get:
            if payload:
                print(
                    "Received defined payload with GET request! Will ignore payload"
                )
            payload = None
        return self._connection.make_request(payload, route, requests_command)  # type: ignore

    def handle_bad_response(
        self,
        endpoint,
        requests_command,
        requests_response=None,
        aiohttp_response=None,
    ):
        self._connection.handle_bad_response(
            endpoint, requests_command, requests_response, aiohttp_response
        )

    def _set_api_key(self, api_key):
        """Fetch API key from environment variable NUCLEUS_API_KEY if not set"""
        api_key = (
            api_key if api_key else os.environ.get("NUCLEUS_API_KEY", None)
        )
        if api_key is None:
            raise NoAPIKey()

        return api_key
