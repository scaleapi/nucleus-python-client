"""
Nucleus Python Library.

For full documentation see: https://dashboard.scale.com/nucleus/docs/api?language=python
"""

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
    "DatasetItem",
    "DatasetItemRetrievalError",
    "Frame",
    "Frame",
    "LidarScene",
    "LidarScene",
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
]

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import aiohttp
import nest_asyncio
import pkg_resources
import requests
import tqdm
import tqdm.notebook as tqdm_notebook

from nucleus.url_utils import sanitize_string_args

from .annotation import (
    BoxAnnotation,
    CuboidAnnotation,
    Point,
    Point3D,
    PolygonAnnotation,
    CategoryAnnotation,
    MultiCategoryAnnotation,
    Segment,
    SegmentationAnnotation,
)
from .constants import (
    ANNOTATION_METADATA_SCHEMA_KEY,
    ANNOTATIONS_IGNORED_KEY,
    ANNOTATIONS_PROCESSED_KEY,
    AUTOTAGS_KEY,
    DATASET_ID_KEY,
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
    NAME_KEY,
    NUCLEUS_ENDPOINT,
    PREDICTIONS_IGNORED_KEY,
    PREDICTIONS_PROCESSED_KEY,
    REFERENCE_IDS_KEY,
    SLICE_ID_KEY,
    STATUS_CODE_KEY,
    UPDATE_KEY,
)
from .dataset import Dataset
from .dataset_item import CameraParams, DatasetItem, Quaternion
from .errors import (
    DatasetItemRetrievalError,
    ModelCreationError,
    ModelRunCreationError,
    NotFoundError,
    NucleusAPIError,
)
from .job import AsyncJob
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
    CuboidPrediction,
    PolygonPrediction,
    SegmentationPrediction,
    CategoryPrediction,
)
from .scene import Frame, LidarScene
from .slice import Slice
from .upload_response import UploadResponse

# pylint: disable=E1101
# TODO: refactor to reduce this file to under 1000 lines.
# pylint: disable=C0302


__version__ = pkg_resources.get_distribution("scale-nucleus").version

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger(requests.packages.urllib3.__package__).setLevel(
    logging.ERROR
)


class RetryStrategy:
    statuses = {503, 504}
    sleep_times = [1, 3, 9]


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
        api_key: str,
        use_notebook: bool = False,
        endpoint: str = None,
    ):
        self.api_key = api_key
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

    def __repr__(self):
        return f"NucleusClient(api_key='{self.api_key}', use_notebook={self._use_notebook}, endpoint='{self.endpoint}')"

    def __eq__(self, other):
        if self.api_key == other.api_key:
            if self._use_notebook == other._use_notebook:
                return True
        return False

    def list_models(self) -> List[Model]:
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
            )
            for model in model_objects["models"]
        ]

    def list_datasets(self) -> Dict[str, Union[str, List[str]]]:
        """Fetches all of your Nucleus datasets.

        Returns:
            Payload containing all dataset IDs associated with the client
            API key.
        """
        return self.make_request({}, "dataset/", requests.get)

    def list_jobs(
        self, show_completed=None, date_limit=None
    ) -> List[AsyncJob]:
        """Fetches all of your running jobs in Nucleus.

        Returns:
            List[:class:`AsyncJob`]: List of running asynchronous jobs
            associated with the client API key.
        """
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

    def get_dataset_items(self, dataset_id) -> List[DatasetItem]:
        # TODO: deprecate in favor of Dataset.items
        response = self.make_request(
            {}, f"dataset/{dataset_id}/datasetItems", requests.get
        )
        dataset_items = response.get("dataset_items", None)
        error = response.get("error", None)
        constructed_dataset_items = []
        if dataset_items:
            for item in dataset_items:
                image_url = item.get("original_image_url")
                metadata = item.get("metadata", None)
                ref_id = item.get("ref_id", None)
                dataset_item = DatasetItem(image_url, ref_id, metadata)
                constructed_dataset_items.append(dataset_item)
        elif error:
            raise DatasetItemRetrievalError(message=error)

        return constructed_dataset_items

    def get_dataset(self, dataset_id: str) -> Dataset:
        """Fetches a dataset by its ID.

        Parameters:
            dataset_id: The ID of the dataset to fetch.

        Returns:
            :class:`Dataset`: The Nucleus dataset as an object.
        """
        return Dataset(dataset_id, self)

    def get_model(self, model_id: str) -> Model:
        """Fetches a model by its ID.

        Parameters:
            model_id: Nucleus-generated model ID (starts with ``prj_``). This can
              be retrieved via :meth:`list_models` or a Nucleus dashboard URL.

        Returns:
            :class:`Model`: The Nucleus model as an object.
        """
        payload = self.make_request(
            payload={},
            route=f"model/{model_id}",
            requests_command=requests.get,
        )
        return Model.from_json(payload=payload, client=self)

    def get_model_run(self, model_run_id: str, dataset_id: str) -> ModelRun:
        # TODO: deprecate ModelRun
        return ModelRun(model_run_id, dataset_id, self)

    def delete_model_run(self, model_run_id: str):
        # TODO: deprecate ModelRun
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
        item_metadata_schema: Optional[Dict] = None,
        annotation_metadata_schema: Optional[Dict] = None,
    ) -> Dataset:
        """
        Creates a new, empty dataset.

        Parameters:
            name: A human-readable name for the dataset.
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
        response = self.make_request(
            {
                NAME_KEY: name,
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

    @sanitize_string_args
    def delete_dataset_item(self, dataset_id: str, reference_id) -> dict:
        # TODO: deprecate in favor of Dataset.delete_item invocation
        return self.make_request(
            {},
            f"dataset/{dataset_id}/refloc/{reference_id}",
            requests.delete,
        )

    def populate_dataset(
        self,
        dataset_id: str,
        dataset_items: List[DatasetItem],
        batch_size: int = 20,
        update: bool = False,
    ):
        # TODO: deprecate in favor of Dataset.append invocation
        local_items = []
        remote_items = []

        # Check local files exist before sending requests
        for item in dataset_items:
            if item.local:
                if not item.local_file_exists():
                    raise NotFoundError()
                local_items.append(item)
            else:
                remote_items.append(item)

        local_batches = [
            local_items[i : i + batch_size]
            for i in range(0, len(local_items), batch_size)
        ]

        remote_batches = [
            remote_items[i : i + batch_size]
            for i in range(0, len(remote_items), batch_size)
        ]

        agg_response = UploadResponse(json={DATASET_ID_KEY: dataset_id})

        async_responses: List[Any] = []

        if local_batches:
            tqdm_local_batches = self.tqdm_bar(
                local_batches, desc="Local file batches"
            )

            for batch in tqdm_local_batches:
                payload = construct_append_payload(batch, update)
                responses = self._process_append_requests_local(
                    dataset_id, payload, update
                )
                async_responses.extend(responses)

        if remote_batches:
            tqdm_remote_batches = self.tqdm_bar(
                remote_batches, desc="Remote file batches"
            )
            for batch in tqdm_remote_batches:
                payload = construct_append_payload(batch, update)
                responses = self._process_append_requests(
                    dataset_id=dataset_id,
                    payload=payload,
                    update=update,
                    batch_size=batch_size,
                )
                async_responses.extend(responses)

        for response in async_responses:
            agg_response.update_response(response)

        return agg_response

    def _process_append_requests_local(
        self,
        dataset_id: str,
        payload: dict,
        update: bool,  # TODO: understand how to pass this in.
        local_batch_size: int = 10,
    ):
        def get_files(batch):
            for item in batch:
                item[UPDATE_KEY] = update
            request_payload = [
                (
                    ITEMS_KEY,
                    (
                        None,
                        json.dumps(batch, allow_nan=False),
                        "application/json",
                    ),
                )
            ]
            for item in batch:
                image = open(  # pylint: disable=R1732
                    item.get(IMAGE_URL_KEY), "rb"  # pylint: disable=R1732
                )  # pylint: disable=R1732
                img_name = os.path.basename(image.name)
                img_type = (
                    f"image/{os.path.splitext(image.name)[1].strip('.')}"
                )
                request_payload.append(
                    (IMAGE_KEY, (img_name, image, img_type))
                )
            return request_payload

        items = payload[ITEMS_KEY]
        responses: List[Any] = []
        files_per_request = []
        payload_items = []
        for i in range(0, len(items), local_batch_size):
            batch = items[i : i + local_batch_size]
            files_per_request.append(get_files(batch))
            payload_items.append(batch)

        future = self.make_many_files_requests_asynchronously(
            files_per_request,
            f"dataset/{dataset_id}/append",
        )

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:  # no event loop running:
            loop = asyncio.new_event_loop()
            responses = loop.run_until_complete(future)
        else:
            nest_asyncio.apply(loop)
            return loop.run_until_complete(future)

        def close_files(request_items):
            for item in request_items:
                # file buffer in location [1][1]
                if item[0] == IMAGE_KEY:
                    item[1][1].close()

        # don't forget to close all open files
        for p in files_per_request:
            close_files(p)

        return responses

    async def make_many_files_requests_asynchronously(
        self, files_per_request, route
    ):
        """Makes an async post request with files to a Nucleus endpoint.

        Parameters:
            files_per_request (List): Nested list of tuples ``(name, (filename,
              file_pointer, content_type))``; multer will build an array by ``name``.
            route (str): Route for the request.

        Returns:
            List: Awaitable ``asyncio`` response.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.ensure_future(
                    self._make_files_request(
                        files=files, route=route, session=session
                    )
                )
                for files in files_per_request
            ]
            return await asyncio.gather(*tasks)

    async def _make_files_request(
        self,
        files,
        route: str,
        session: aiohttp.ClientSession,
        retry_attempt=0,
        max_retries=3,
        sleep_intervals=(1, 3, 9),
    ):
        """Makes an async post request with files to a Nucleus endpoint.

        Parameters:
            files (List[Tuple]): A list of tuples ``(name, (filename, file_pointer, file_type))``.
            route: Route for the request.
            session: Session to use for post.

        Returns:
            Awaitable ``asyncio`` JSON response.
        """
        endpoint = f"{self.endpoint}/{route}"

        logger.info("Posting to %s", endpoint)

        form = aiohttp.FormData()

        for file in files:
            form.add_field(
                name=file[0],
                filename=file[1][0],
                value=file[1][1],
                content_type=file[1][2],
            )

        for sleep_time in RetryStrategy.sleep_times + [-1]:

            async with session.post(
                endpoint,
                data=form,
                auth=aiohttp.BasicAuth(self.api_key, ""),
                timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
            ) as response:
                logger.info(
                    "API request has response code %s", response.status
                )

                try:
                    data = await response.json()
                except aiohttp.client_exceptions.ContentTypeError:
                    # In case of 404, the server returns text
                    data = await response.text()
                if (
                    response.status in RetryStrategy.statuses
                    and sleep_time != -1
                ):
                    time.sleep(sleep_time)
                    continue

                if not response.ok:
                    if retry_attempt < max_retries:
                        time.sleep(sleep_intervals[retry_attempt])
                        retry_attempt += 1
                        return self._make_files_request(
                            files,
                            route,
                            session,
                            retry_attempt,
                            max_retries,
                            sleep_intervals,
                        )
                    else:
                        self.handle_bad_response(
                            endpoint,
                            session.post,
                            aiohttp_response=(
                                response.status,
                                response.reason,
                                data,
                            ),
                        )

                return data

    def _process_append_requests(
        self,
        dataset_id: str,
        payload: dict,
        update: bool,
        batch_size: int = 20,
    ):
        items = payload[ITEMS_KEY]
        payloads = [
            # batch_size images per request
            {ITEMS_KEY: items[i : i + batch_size], UPDATE_KEY: update}
            for i in range(0, len(items), batch_size)
        ]

        return [
            self.make_request(
                payload,
                f"dataset/{dataset_id}/append",
            )
            for payload in payloads
        ]

    def annotate_dataset(
        self,
        dataset_id: str,
        annotations: List[
            Union[
                BoxAnnotation,
                PolygonAnnotation,
                CuboidAnnotation,
                CategoryAnnotation,
                MultiCategoryAnnotation,
                SegmentationAnnotation,
            ]
        ],
        update: bool,
        batch_size: int = 5000,
    ) -> Dict[str, Union[str, int]]:
        # TODO: deprecate in favor of Dataset.annotate invocation

        # Split payload into segmentations and Box/Polygon
        segmentations = [
            ann
            for ann in annotations
            if isinstance(ann, SegmentationAnnotation)
        ]
        other_annotations = [
            ann
            for ann in annotations
            if not isinstance(ann, SegmentationAnnotation)
        ]

        batches = [
            other_annotations[i : i + batch_size]
            for i in range(0, len(other_annotations), batch_size)
        ]

        semseg_batches = [
            segmentations[i : i + batch_size]
            for i in range(0, len(segmentations), batch_size)
        ]

        agg_response = {
            DATASET_ID_KEY: dataset_id,
            ANNOTATIONS_PROCESSED_KEY: 0,
            ANNOTATIONS_IGNORED_KEY: 0,
        }

        total_batches = len(batches) + len(semseg_batches)

        tqdm_batches = self.tqdm_bar(batches)

        with self.tqdm_bar(total=total_batches) as pbar:
            for batch in tqdm_batches:
                payload = construct_annotation_payload(batch, update)
                response = self.make_request(
                    payload, f"dataset/{dataset_id}/annotate"
                )
                pbar.update(1)
                if STATUS_CODE_KEY in response:
                    agg_response[ERRORS_KEY] = response
                else:
                    agg_response[ANNOTATIONS_PROCESSED_KEY] += response[
                        ANNOTATIONS_PROCESSED_KEY
                    ]
                    agg_response[ANNOTATIONS_IGNORED_KEY] += response[
                        ANNOTATIONS_IGNORED_KEY
                    ]

            for s_batch in semseg_batches:
                payload = construct_segmentation_payload(s_batch, update)
                response = self.make_request(
                    payload, f"dataset/{dataset_id}/annotate_segmentation"
                )
                pbar.update(1)
                if STATUS_CODE_KEY in response:
                    agg_response[ERRORS_KEY] = response
                else:
                    agg_response[ANNOTATIONS_PROCESSED_KEY] += response[
                        ANNOTATIONS_PROCESSED_KEY
                    ]
                    agg_response[ANNOTATIONS_IGNORED_KEY] += response[
                        ANNOTATIONS_IGNORED_KEY
                    ]

        return agg_response

    def ingest_tasks(self, dataset_id: str, payload: dict):
        # TODO: deprecate in favor of Dataset.ingest_tasks invocation
        return self.make_request(payload, f"dataset/{dataset_id}/ingest_tasks")

    def add_model(
        self, name: str, reference_id: str, metadata: Optional[Dict] = None
    ) -> Model:
        # TODO: consistency between ``add`` vs. ``create``
        """Adds a :class:`Model` to Nucleus.

        Parameters:
            name: A human-readable name for the model.
            reference_id: Unique, user-controlled ID for the model. This can be
              used, for example, to link to an external storage of models which
              may have its own id scheme.
            metadata: An arbitrary dictionary of additional data about this model
              that can be stored and retrieved. For example, you can store information
              about the hyperparameters used in training this model.

        Returns:
            :class:`Model`: The newly created model as an object.
        """
        response = self.make_request(
            construct_model_creation_payload(name, reference_id, metadata),
            "models/add",
        )
        model_id = response.get("model_id", None)
        if not model_id:
            raise ModelCreationError(response.get("error"))

        return Model(model_id, name, reference_id, metadata, self)

    def create_model_run(self, dataset_id: str, payload: dict) -> ModelRun:
        # TODO: deprecate ModelRun
        response = self.make_request(
            payload, f"dataset/{dataset_id}/modelRun/create"
        )
        if response.get(STATUS_CODE_KEY, None):
            raise ModelRunCreationError(response.get("error"))

        return ModelRun(
            response[MODEL_RUN_ID_KEY], dataset_id=dataset_id, client=self
        )

    def predict(
        self,
        annotations: List[
            Union[
                BoxPrediction,
                PolygonPrediction,
                CuboidPrediction,
                SegmentationPrediction,
                CategoryPrediction,
            ]
        ],
        model_run_id: Optional[str] = None,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        update: bool = False,
        batch_size: int = 5000,
    ):
        # TODO: deprecate in favor of Dataset.upload_predictions invocation
        if model_run_id is not None:
            assert model_id is None and dataset_id is None
            endpoint = f"modelRun/{model_run_id}/predict"
        else:
            assert (
                model_id is not None and dataset_id is not None
            ), "Model ID and dataset ID are required if not using model run id."
            endpoint = (
                f"dataset/{dataset_id}/model/{model_id}/uploadPredictions"
            )
        segmentations = [
            ann
            for ann in annotations
            if isinstance(ann, SegmentationPrediction)
        ]

        other_predictions = [
            ann
            for ann in annotations
            if not isinstance(ann, SegmentationPrediction)
        ]

        s_batches = [
            segmentations[i : i + batch_size]
            for i in range(0, len(segmentations), batch_size)
        ]

        batches = [
            other_predictions[i : i + batch_size]
            for i in range(0, len(other_predictions), batch_size)
        ]

        errors = []
        predictions_processed = 0
        predictions_ignored = 0

        tqdm_batches = self.tqdm_bar(batches)

        for batch in tqdm_batches:
            batch_payload = construct_box_predictions_payload(
                batch,
                update,
            )
            response = self.make_request(batch_payload, endpoint)
            if STATUS_CODE_KEY in response:
                errors.append(response)
            else:
                predictions_processed += response[PREDICTIONS_PROCESSED_KEY]
                predictions_ignored += response[PREDICTIONS_IGNORED_KEY]

        for s_batch in s_batches:
            payload = construct_segmentation_payload(s_batch, update)
            response = self.make_request(payload, endpoint)
            # pbar.update(1)
            if STATUS_CODE_KEY in response:
                errors.append(response)
            else:
                predictions_processed += response[PREDICTIONS_PROCESSED_KEY]
                predictions_ignored += response[PREDICTIONS_IGNORED_KEY]

        return {
            MODEL_RUN_ID_KEY: model_run_id,
            PREDICTIONS_PROCESSED_KEY: predictions_processed,
            PREDICTIONS_IGNORED_KEY: predictions_ignored,
            ERRORS_KEY: errors,
        }

    def commit_model_run(
        self, model_run_id: str, payload: Optional[dict] = None
    ):
        # TODO: deprecate ModelRun. this should be renamed to calculate_evaluation_metrics
        #   or completely removed in favor of Model class methods
        if payload is None:
            payload = {}
        return self.make_request(payload, f"modelRun/{model_run_id}/commit")

    def dataset_info(self, dataset_id: str):
        # TODO: deprecate in favor of Dataset.info invocation
        return self.make_request(
            {}, f"dataset/{dataset_id}/info", requests.get
        )

    def model_run_info(self, model_run_id: str):
        # TODO: deprecate ModelRun
        return self.make_request(
            {}, f"modelRun/{model_run_id}/info", requests.get
        )

    @sanitize_string_args
    def dataitem_ref_id(self, dataset_id: str, reference_id: str):
        # TODO: deprecate in favor of Dataset.refloc invocation
        return self.make_request(
            {}, f"dataset/{dataset_id}/refloc/{reference_id}", requests.get
        )

    @sanitize_string_args
    def predictions_ref_id(self, model_run_id: str, ref_id: str):
        # TODO: deprecate ModelRun
        return self.make_request(
            {}, f"modelRun/{model_run_id}/refloc/{ref_id}", requests.get
        )

    def dataitem_iloc(self, dataset_id: str, i: int):
        # TODO: deprecate in favor of Dataset.iloc invocation
        return self.make_request(
            {}, f"dataset/{dataset_id}/iloc/{i}", requests.get
        )

    def predictions_iloc(self, model_run_id: str, i: int):
        # TODO: deprecate ModelRun
        return self.make_request(
            {}, f"modelRun/{model_run_id}/iloc/{i}", requests.get
        )

    def dataitem_loc(self, dataset_id: str, dataset_item_id: str):
        # TODO: deprecate in favor of Dataset.loc invocation
        return self.make_request(
            {}, f"dataset/{dataset_id}/loc/{dataset_item_id}", requests.get
        )

    def predictions_loc(self, model_run_id: str, dataset_item_id: str):
        # TODO: deprecate ModelRun
        return self.make_request(
            {}, f"modelRun/{model_run_id}/loc/{dataset_item_id}", requests.get
        )

    def create_slice(self, dataset_id: str, payload: dict) -> Slice:
        # TODO: deprecate in favor of Dataset.create_slice
        response = self.make_request(
            payload, f"dataset/{dataset_id}/create_slice"
        )
        return Slice(response[SLICE_ID_KEY], self)

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

    def slice_info(self, slice_id: str) -> dict:
        # TODO: migrate to Slice method and deprecate
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

    def delete_annotations(
        self, dataset_id: str, reference_ids: list = None, keep_history=False
    ) -> dict:
        # TODO: deprecate in favor of Dataset.delete_annotations invocation
        payload = {KEEP_HISTORY_KEY: keep_history}
        if reference_ids:
            payload[REFERENCE_IDS_KEY] = reference_ids
        response = self.make_request(
            payload,
            f"annotation/{dataset_id}",
            requests_command=requests.delete,
        )
        return response

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

    def create_custom_index(
        self, dataset_id: str, embeddings_urls: list, embedding_dim: int
    ):
        # TODO: deprecate in favor of Dataset.create_custom_index invocation
        return self.make_request(
            {
                EMBEDDINGS_URL_KEY: embeddings_urls,
                EMBEDDING_DIMENSION_KEY: embedding_dim,
            },
            f"indexing/{dataset_id}",
            requests_command=requests.post,
        )

    def delete_custom_index(self, dataset_id: str):
        # TODO: deprecate in favor of Dataset.delete_custom_index invocation
        return self.make_request(
            {},
            f"indexing/{dataset_id}",
            requests_command=requests.delete,
        )

    def set_continuous_indexing(self, dataset_id: str, enable: bool = True):
        # TODO: deprecate in favor of Dataset.set_continuous_indexing invocation
        return self.make_request(
            {INDEX_CONTINUOUS_ENABLE_KEY: enable},
            f"indexing/{dataset_id}/setContinuous",
            requests_command=requests.post,
        )

    def create_image_index(self, dataset_id: str):
        # TODO: deprecate in favor of Dataset.set_continuous_indexing invocation
        return self.make_request(
            {},
            f"indexing/{dataset_id}/internal/image",
            requests_command=requests.post,
        )

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

    def make_request(
        self, payload: dict, route: str, requests_command=requests.post
    ) -> dict:
        """Makes a request to a Nucleus API endpoint.

        Logs a warning if not successful.

        Parameters:
            payload: Given request payload.
            route: Route for the request.
            Requests command: ``requests.post``, ``requests.get``, or ``requests.delete``.

        Returns:
            Response payload as JSON-like dict.
        """
        endpoint = f"{self.endpoint}/{route}"

        logger.info("Posting to %s", endpoint)

        for retry_wait_time in RetryStrategy.sleep_times:
            response = requests_command(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                auth=(self.api_key, ""),
                timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
            )
            logger.info(
                "API request has response code %s", response.status_code
            )
            if response.status_code not in RetryStrategy.statuses:
                break
            time.sleep(retry_wait_time)

        if not response.ok:
            self.handle_bad_response(endpoint, requests_command, response)

        return response.json()

    def handle_bad_response(
        self,
        endpoint,
        requests_command,
        requests_response=None,
        aiohttp_response=None,
    ):
        raise NucleusAPIError(
            endpoint, requests_command, requests_response, aiohttp_response
        )
