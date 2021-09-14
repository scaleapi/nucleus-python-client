"""
Nucleus Python Library.

For full documentation see: https://dashboard.scale.com/nucleus/docs/api?language=python
"""
import asyncio
import json
import logging
import os
import urllib.request
from asyncio.tasks import Task
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
    JOB_ID_KEY,
    JOB_LAST_KNOWN_STATUS_KEY,
    JOB_TYPE_KEY,
    JOB_CREATION_TIME_KEY,
    IMAGE_KEY,
    IMAGE_URL_KEY,
    INDEX_CONTINUOUS_ENABLE_KEY,
    ITEM_METADATA_SCHEMA_KEY,
    ITEMS_KEY,
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
from .dataset_item import DatasetItem, CameraParams, Quaternion
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
)
from .slice import Slice
from .upload_response import UploadResponse
from .scene import Frame, LidarScene

# pylint: disable=E1101
# TODO: refactor to reduce this file to under 1000 lines.
# pylint: disable=C0302


__version__ = pkg_resources.get_distribution("scale-nucleus").version

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger(requests.packages.urllib3.__package__).setLevel(
    logging.ERROR
)


class NucleusClient:
    """
    Nucleus client.
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
        """
        Lists available models in your repo.
        :return: model_ids
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
        """
        Lists available datasets in your repo.
        :return: { datasets_ids }
        """
        return self.make_request({}, "dataset/", requests.get)

    def list_jobs(
        self, show_completed=None, date_limit=None
    ) -> List[AsyncJob]:
        """
        Lists jobs for user.
        :return: jobs
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
        """
        Gets all the dataset items inside your repo as a json blob.
        :return [ DatasetItem ]
        """
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
        """
        Fetches a dataset for given id
        :param dataset_id: internally controlled dataset_id
        :return: dataset
        """
        return Dataset(dataset_id, self)

    def get_model(self, model_id: str) -> Model:
        """
        Fetched a model for a given id
        :param model_id: internally controlled dataset_id
        :return: model
        """
        payload = self.make_request(
            payload={},
            route=f"model/{model_id}",
            requests_command=requests.get,
        )
        return Model.from_json(payload=payload, client=self)

    def get_model_run(self, model_run_id: str, dataset_id: str) -> ModelRun:
        """
        Fetches a model_run for given id
        :param model_run_id: internally controlled model_run_id
        :param dataset_id: the dataset id which may determine the prediction schema
            for this model run if present on the dataset.
        :return: model_run
        """
        return ModelRun(model_run_id, dataset_id, self)

    def delete_model_run(self, model_run_id: str):
        """
        Fetches a model_run for given id
        :param model_run_id: internally controlled model_run_id
        :return: model_run
        """
        return self.make_request(
            {}, f"modelRun/{model_run_id}", requests.delete
        )

    def create_dataset_from_project(
        self, project_id: str, last_n_tasks: int = None, name: str = None
    ) -> Dataset:
        """
        Creates a new dataset based on payload params:
        name -- A human-readable name of the dataset.
        Returns a response with internal id and name for a new dataset.
        :param payload: { "name": str }
        :return: new Dataset object
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
        Creates a new dataset:
        Returns a response with internal id and name for a new dataset.
        :param name -- A human-readable name of the dataset.
        :param item_metadata_schema -- optional dictionary to define item metadata schema
        :param annotation_metadata_schema -- optional dictionary to define annotation metadata schema
        :return: new Dataset object
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
        Deletes a private dataset based on datasetId.
        Returns an empty payload where response status `200` indicates
        the dataset has been successfully deleted.
        :param payload: { "name": str }
        :return: { "dataset_id": str, "name": str }
        """
        return self.make_request({}, f"dataset/{dataset_id}", requests.delete)

    @sanitize_string_args
    def delete_dataset_item(self, dataset_id: str, reference_id) -> dict:
        """
        Deletes a private dataset based on datasetId.
        Returns an empty payload where response status `200` indicates
        the dataset has been successfully deleted.
        :param payload: { "name": str }
        :return: { "dataset_id": str, "name": str }
        """
        return self.make_request(
            {},
            f"dataset/{dataset_id}/refloc/{reference_id}",
            requests.delete,
        )

    def populate_dataset(
        self,
        dataset_id: str,
        dataset_items: List[DatasetItem],
        batch_size: int = 100,
        update: bool = False,
    ):
        """
        Appends images to a dataset with given dataset_id.
        Overwrites images on collision if updated.
        :param dataset_id: id of a dataset
        :param payload: { "items": List[DatasetItem], "update": bool }
        :param local: flag if images are stored locally
        :param batch_size: size of the batch for long payload
        :return:
        {
            "dataset_id: str,
            "new_items": int,
            "updated_items": int,
            "ignored_items": int,
            "upload_errors": int
        }
        """
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
        """
        Makes an async post request with files to a Nucleus endpoint.

        :param files_per_request: A list of lists of tuples (name, (filename, file_pointer, content_type))
           name will become the name by which the multer can build an array.
        :param route: route for the request
        :return: awaitable list(response)
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
    ):
        """
        Makes an async post request with files to a Nucleus endpoint.

        :param files: A list of tuples (name, (filename, file_pointer, file_type))
        :param route: route for the request
        :param session: Session to use for post.
        :return: response
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

        async with session.post(
            endpoint,
            data=form,
            auth=aiohttp.BasicAuth(self.api_key, ""),
            timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
        ) as response:
            logger.info("API request has response code %s", response.status)

            try:
                data = await response.json()
            except aiohttp.client_exceptions.ContentTypeError:
                # In case of 404, the server returns text
                data = await response.text()

            if not response.ok:
                self.handle_bad_response(
                    endpoint,
                    session.post,
                    aiohttp_response=(response.status, response.reason, data),
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
                SegmentationAnnotation,
            ]
        ],
        update: bool,
        batch_size: int = 5000,
    ):
        """
        Uploads ground truth annotations for a given dataset.
        :param dataset_id: id of the dataset
        :param annotations: List[Union[BoxAnnotation, PolygonAnnotation, CuboidAnnotation, SegmentationAnnotation]]
        :param update: whether to update or ignore conflicting annotations
        :return: {"dataset_id: str, "annotations_processed": int}
        """
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
        """
        If you already submitted tasks to Scale for annotation this endpoint ingests your completed tasks
        annotated by Scale into your Nucleus Dataset.
        Right now we support ingestion from Videobox Annotation and 2D Box Annotation projects.
        :param payload: {"tasks" : List[task_ids]}
        :param dataset_id: id of the dataset
        :return: {"ingested_tasks": int, "ignored_tasks": int, "pending_tasks": int}
        """
        return self.make_request(payload, f"dataset/{dataset_id}/ingest_tasks")

    def add_model(
        self, name: str, reference_id: str, metadata: Optional[Dict] = None
    ) -> Model:
        """
        Adds a model info to your repo based on payload params:
        name -- A human-readable name of the model project.
        reference_id -- An optional user-specified identifier to reference this given model.
        metadata -- An arbitrary metadata blob for the model.
        :param name: A human-readable name of the model project.
        :param reference_id: An user-specified identifier to reference this given model.
        :param metadata: An optional arbitrary metadata blob for the model.
        :return: { "model_id": str }
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
        """
        Creates model run for dataset_id based on the given parameters specified in the payload:

        'reference_id' -- The user-specified reference identifier to associate with the model.
                        The 'model_id' field should be empty if this field is populated.

        'model_id' -- The internally-controlled identifier of the model.
                    The 'reference_id' field should be empty if this field is populated.

        'name' -- An optional name for the model run.

        'metadata' -- An arbitrary metadata blob for the current run.

        :param
        dataset_id: id of the dataset
        payload:
        {
            "reference_id": str,
            "model_id": str,
            "name": Optional[str],
            "metadata": Optional[Dict[str, Any]],
        }
        :return: new ModelRun object
        """
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
        model_run_id: str,
        annotations: List[
            Union[
                BoxPrediction,
                PolygonPrediction,
                CuboidPrediction,
                SegmentationPrediction,
            ]
        ],
        update: bool,
        batch_size: int = 5000,
    ):
        """
        Uploads model outputs as predictions for a model_run. Returns info about the upload.
        :param annotations: List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        :param update: bool
        :return:
        {
            "dataset_id": str,
            "model_run_id": str,
            "predictions_processed": int,
            "predictions_ignored": int,
        }
        """
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

        agg_response = {
            MODEL_RUN_ID_KEY: model_run_id,
            PREDICTIONS_PROCESSED_KEY: 0,
            PREDICTIONS_IGNORED_KEY: 0,
        }

        tqdm_batches = self.tqdm_bar(batches)

        for batch in tqdm_batches:
            batch_payload = construct_box_predictions_payload(
                batch,
                update,
            )
            response = self.make_request(
                batch_payload, f"modelRun/{model_run_id}/predict"
            )
            if STATUS_CODE_KEY in response:
                agg_response[ERRORS_KEY] = response
            else:
                agg_response[PREDICTIONS_PROCESSED_KEY] += response[
                    PREDICTIONS_PROCESSED_KEY
                ]
                agg_response[PREDICTIONS_IGNORED_KEY] += response[
                    PREDICTIONS_IGNORED_KEY
                ]

        for s_batch in s_batches:
            payload = construct_segmentation_payload(s_batch, update)
            response = self.make_request(
                payload, f"modelRun/{model_run_id}/predict_segmentation"
            )
            # pbar.update(1)
            if STATUS_CODE_KEY in response:
                agg_response[ERRORS_KEY] = response
            else:
                agg_response[PREDICTIONS_PROCESSED_KEY] += response[
                    PREDICTIONS_PROCESSED_KEY
                ]
                agg_response[PREDICTIONS_IGNORED_KEY] += response[
                    PREDICTIONS_IGNORED_KEY
                ]

        return agg_response

    def commit_model_run(
        self, model_run_id: str, payload: Optional[dict] = None
    ):
        """
        Commits the model run. Starts matching algorithm defined by payload.
        class_agnostic -- A flag to specify if matching algorithm should be class-agnostic or not.
                          Default value: True

        allowed_label_matches -- An optional list of AllowedMatch objects to specify allowed matches
                                 for ground truth and model predictions.
                                 If specified, 'class_agnostic' flag is assumed to be False

        Type 'AllowedMatch':
        {
            ground_truth_label: string,       # A label for ground truth annotation.
            model_prediction_label: string,   # A label for model prediction that can be matched with
                                              # corresponding ground truth label.
        }

        payload:
        {
            "class_agnostic": boolean,
            "allowed_label_matches": List[AllowedMatch],
        }

        :return: {"model_run_id": str}
        """
        if payload is None:
            payload = {}
        return self.make_request(payload, f"modelRun/{model_run_id}/commit")

    def dataset_info(self, dataset_id: str):
        """
        Returns information about existing dataset
        :param dataset_id: dataset id
        :return: dictionary of the form
            {
                'name': str,
                'length': int,
                'model_run_ids': List[str],
                'slice_ids': List[str]
            }
        """
        return self.make_request(
            {}, f"dataset/{dataset_id}/info", requests.get
        )

    def model_run_info(self, model_run_id: str):
        """
        provides information about a Model Run with given model_run_id:
        model_id -- Model Id corresponding to the run
        name -- A human-readable name of the model project.
        status -- Status of the Model Run.
        metadata -- An arbitrary metadata blob specified for the run.
        :return:
        {
            "model_id": str,
            "name": str,
            "status": str,
            "metadata": Dict[str, Any],
        }
        """
        return self.make_request(
            {}, f"modelRun/{model_run_id}/info", requests.get
        )

    @sanitize_string_args
    def dataitem_ref_id(self, dataset_id: str, reference_id: str):
        """
        :param dataset_id: internally controlled dataset id
        :param reference_id: reference_id of a dataset_item
        :return:
        """
        return self.make_request(
            {}, f"dataset/{dataset_id}/refloc/{reference_id}", requests.get
        )

    @sanitize_string_args
    def predictions_ref_id(self, model_run_id: str, ref_id: str):
        """
        Returns Model Run info For Dataset Item by model_run_id and item reference_id.
        :param model_run_id: id of the model run.
        :param reference_id: reference_id of a dataset item.
        :return:
        {
            "annotations": List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        }
        """
        return self.make_request(
            {}, f"modelRun/{model_run_id}/refloc/{ref_id}", requests.get
        )

    def dataitem_iloc(self, dataset_id: str, i: int):
        """
        Returns Dataset Item info by dataset_id and absolute number of the dataset item.
        :param dataset_id:  internally controlled dataset id
        :param i: absolute number of the dataset_item
        :return:
        """
        return self.make_request(
            {}, f"dataset/{dataset_id}/iloc/{i}", requests.get
        )

    def predictions_iloc(self, model_run_id: str, i: int):
        """
        Returns Model Run Info For Dataset Item by model_run_id and absolute number of an item.
        :param model_run_id: id of the model run.
        :param i: absolute number of Dataset Item for a dataset corresponding to the model run.
        :return:
        {
            "annotations": List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        }
        """
        return self.make_request(
            {}, f"modelRun/{model_run_id}/iloc/{i}", requests.get
        )

    def dataitem_loc(self, dataset_id: str, dataset_item_id: str):
        """
        Returns Dataset Item Info By dataset_item_id and dataset_id
        :param dataset_id: internally controlled id for the dataset.
        :param dataset_item_id: internally controlled id for the dataset item.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Box2DAnnotation],
        }
        """
        return self.make_request(
            {}, f"dataset/{dataset_id}/loc/{dataset_item_id}", requests.get
        )

    def predictions_loc(self, model_run_id: str, dataset_item_id: str):
        """
        Returns Model Run Info For Dataset Item by its id.
        :param model_run_id: id of the model run.
        :param dataset_item_id: dataset_item_id of a dataset item.
        :return:
        {
            "annotations": List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        }
        """
        return self.make_request(
            {}, f"modelRun/{model_run_id}/loc/{dataset_item_id}", requests.get
        )

    def create_slice(self, dataset_id: str, payload: dict) -> Slice:
        """
        Creates a slice from items already present in a dataset.
        The caller must exclusively use either datasetItemIds or reference_ids
        as a means of identifying items in the dataset.

        "name" -- The human-readable name of the slice.
        "reference_ids" -- An optional list of user-specified identifier for the items in the slice

        :param
        dataset_id: id of the dataset
        payload:
        {
            "name": str,
            "reference_ids": List[str],
        }
        :return: new Slice object
        """
        response = self.make_request(
            payload, f"dataset/{dataset_id}/create_slice"
        )
        return Slice(response[SLICE_ID_KEY], self)

    def get_slice(self, slice_id: str) -> Slice:
        """
        Returns a slice object by specified id.

        :param
        slice_id: id of the slice
        :return: a Slice object
        """
        return Slice(slice_id, self)

    def slice_info(self, slice_id: str) -> dict:
        """
        This endpoint provides information about specified slice.

        :param
        slice_id: id of the slice

        :return:
        {
            "name": str,
            "dataset_id": str,
            "reference_ids": List[str],
        }
        """
        response = self.make_request(
            {},
            f"slice/{slice_id}",
            requests_command=requests.get,
        )
        return response

    def delete_slice(self, slice_id: str) -> dict:
        """
        This endpoint deletes specified slice.

        :param
        slice_id: id of the slice

        :return:
        {}
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
        """
        This endpoint deletes annotations.

        :param
        slice_id: id of the slice

        :return:
        {}
        """
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
        """
        Appends to a slice from items already present in a dataset.
        The caller must exclusively use either datasetItemIds or reference_ids
        as a means of identifying items in the dataset.

        :param
        reference_ids: List[str],

        :return:
        {
            "slice_id": str,
        }
        """

        response = self.make_request(
            {REFERENCE_IDS_KEY: reference_ids}, f"slice/{slice_id}/append"
        )
        return response

    def list_autotags(self, dataset_id: str) -> List[str]:
        """
        Fetches a list of autotags for a given dataset id
        :param dataset_id: internally controlled dataset_id
        :return: List[str] representing autotag_ids
        """
        response = self.make_request(
            {},
            f"{dataset_id}/list_autotags",
            requests_command=requests.get,
        )
        return response[AUTOTAGS_KEY] if AUTOTAGS_KEY in response else response

    def delete_autotag(self, autotag_id: str) -> dict:
        """
        Deletes an autotag based on autotagId.
        Returns an empty payload where response status `200` indicates
        the autotag has been successfully deleted.
        :param autotag_id: id of the autotag to delete.
        :return: {}
        """
        return self.make_request({}, f"autotag/{autotag_id}", requests.delete)

    def delete_model(self, model_id: str) -> dict:
        """
        This endpoint deletes the specified model, along with all
        associated model_runs.

        :param
        model_id: id of the model_run to delete.

        :return:
        {}
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
        """
        Creates a custom index for a given dataset, which will then be used
        for autotag and similarity search.

        :param
        dataset_id: id of dataset that the custom index is being added to.
        embeddings_urls: list of urls, each of which being a json mapping reference_id -> embedding vector
        embedding_dim: the dimension of the embedding vectors, must be consistent for all embedding vectors in the index.
        """
        return self.make_request(
            {
                EMBEDDINGS_URL_KEY: embeddings_urls,
                EMBEDDING_DIMENSION_KEY: embedding_dim,
            },
            f"indexing/{dataset_id}",
            requests_command=requests.post,
        )

    def check_index_status(self, job_id: str):
        return self.make_request(
            {},
            f"indexing/{job_id}",
            requests_command=requests.get,
        )

    def delete_custom_index(self, dataset_id: str):
        return self.make_request(
            {},
            f"indexing/{dataset_id}",
            requests_command=requests.delete,
        )

    def set_continuous_indexing(self, dataset_id: str, enable: bool = True):
        """
        Sets continuous indexing for a given dataset, which will automatically generate embeddings whenever
        new images are uploaded. This endpoint is currently only enabled for enterprise customers.
        Please reach out to nucleus@scale.com if you wish to learn more.

        :param
        dataset_id: id of dataset that continuous indexing is being toggled for
        enable: boolean, sets whether we are enabling or disabling continuous indexing. The default behavior is to enable.
        """
        return self.make_request(
            {INDEX_CONTINUOUS_ENABLE_KEY: enable},
            f"indexing/{dataset_id}/setContinuous",
            requests_command=requests.post,
        )

    def create_image_index(self, dataset_id: str):
        """
        Starts generating embeddings for images that don't have embeddings in a given dataset. These embeddings will
        be used for autotag and similarity search. This endpoint is currently only enabled for enterprise customers.
        Please reach out to nucleus@scale.com if you wish to learn more.

        :param
        dataset_id: id of dataset for generating embeddings on.
        """
        return self.make_request(
            {},
            f"indexing/{dataset_id}/internal/image",
            requests_command=requests.post,
        )

    def make_request(
        self, payload: dict, route: str, requests_command=requests.post
    ) -> dict:
        """
        Makes a request to Nucleus endpoint and logs a warning if not
        successful.

        :param payload: given payload
        :param route: route for the request
        :param requests_command: requests.post, requests.get, requests.delete
        :return: response JSON
        """
        endpoint = f"{self.endpoint}/{route}"

        logger.info("Posting to %s", endpoint)

        response = requests_command(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
            timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
        )
        logger.info("API request has response code %s", response.status_code)

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
