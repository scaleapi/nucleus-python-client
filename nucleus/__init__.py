"""
Nucleus Python Library.

Data formats used:

_____________________________________________________________________________________________________

DatasetItem

image_url       |   str     |   The URL containing the image for the given row of data.\n
reference_id    |   str     |   An optional user-specified identifier to reference this given image.\n
metadata        |   dict    |   All of column definitions for this item.
                |           |   The keys should match the user-specified column names,
                |           |   and the corresponding values will populate the cell under the column.\n
_____________________________________________________________________________________________________


Box2DGeometry:

x               |   float   |   The distance, in pixels, between the left border of the bounding box
                |           |   and the left border of the image.\n
y               |   float   |   The distance, in pixels, between the top border of the bounding box
                |           |   and the top border of the image.\n
width	        |   float   |   The width in pixels of the annotation.\n
height	        |   float   |   The height in pixels of the annotation.\n

Box2DAnnotation:

item_id         |   str     |   The internally-controlled item identifier to associate this annotation with.
                |           |   The reference_id field should be empty if this field is populated.\n
reference_id    |   str     |   The user-specified reference identifier to associate this annotation with.\n
                |           |   The item_id field should be empty if this field is populated.
label	        |   str     |	The label for this annotation (e.g. car, pedestrian, bicycle).\n
type	        |   str     |   The type of this annotation. It should always be the box string literal.\n
geometry        |   dict    |   Representation of the bounding box in the Box2DGeometry format.\n
metadata        |   dict    |   An arbitrary metadata blob for the annotation.\n

_____________________________________________________________________________________________________

Box2DDetection:

item_id         |   str     |   The internally-controlled item identifier to associate this annotation with.
                |           |   The reference_id field should be empty if this field is populated.\n
reference_id    |   str     |   The user-specified reference identifier to associate this annotation with.
                |           |   The item_id field should be empty if this field is populated.\n
label	        |   str     |	The label for this annotation (e.g. car, pedestrian, bicycle).\n
type	        |   str     |   The type of this annotation. It should always be the box string literal.\n
confidence      |   float   |   The optional confidence level of this annotation.
                |           |   It should be between 0 and 1 (inclusive).\n
geometry        |   dict    |   Representation of the bounding box in the Box2DGeometry format.\n
metadata        |   dict    |   An arbitrary metadata blob for the annotation.\n
"""

import json
import logging
import os
from typing import List, Union, Dict, Callable, Any, Optional

from .dataset_item import DatasetItem
import tqdm

import grequests
import requests

from .dataset import Dataset
from .annotation import BoxAnnotation
from .prediction import BoxPrediction
from .model_run import ModelRun
from .slice import Slice
from .upload_response import UploadResponse
from .payload_constructor import (
    construct_append_payload,
    construct_box_annotation_payload,
    construct_model_creation_payload,
)
from .constants import (
    NUCLEUS_ENDPOINT,
    ERROR_ITEMS,
    ITEMS_KEY,
    ITEM_KEY,
    IMAGE_KEY,
    IMAGE_URL_KEY,
    DATASET_ID_KEY,
    MODEL_RUN_ID_KEY,
    DATASET_ITEM_ID_KEY,
    SLICE_ID_KEY,
    ANNOTATIONS_PROCESSED_KEY,
    STATUS_CODE_KEY,
)
from .model import Model
from .errors import ModelCreationError, ModelRunCreationError, DatasetItemRetrievalError

logger = logging.getLogger(__name__)
logging.basicConfig()


class NucleusClient:
    """
    Nucleus client.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def list_models(self) -> List[str]:
        """
        Lists available models in your repo.
        :return: model_ids
        """
        # TODO implement API
        raise NotImplementedError

    def list_datasets(self) -> Dict[str, Union[str, List[str]]]:
        """
        Lists available datasets in your repo.
        :return: { datasets_ids }
        """
        return self._make_request({}, "dataset/", requests.get)

    def get_dataset_items(self, dataset_id) -> List[DatasetItem]:
        """
        Gets all the dataset items inside your repo as a json blob.
        :return [ DatasetItem ]
        """
        response = self._make_request({}, f'dataset/{dataset_id}/datasetItems')
        dataset_items = response.get("dataset_items", None)
        error = response.get("error", None)
        constructed_dataset_items = []
        if dataset_items:
            for item in dataset_items:
                image_url = item.get("original_image_url")
                metadata = item.get("metadata")
                item_id = item.get("id")
                ref_id = item.get("ref_id")
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

    def get_model_run(self, model_run_id: str) -> ModelRun:
        """
        Fetches a model_run for given id
        :param model_run_id: internally controlled model_run_id
        :return: model_run
        """
        return ModelRun(model_run_id, self)

    def delete_model_run(self, model_run_id: str):
        """
        Fetches a model_run for given id
        :param model_run_id: internally controlled model_run_id
        :return: model_run
        """
        return self._make_request(
            {}, f"modelRun/{model_run_id}", requests.delete
        )

    def create_dataset(self, name: str) -> Dataset:
        """
        Creates a new dataset based on payload params:
        name -- A human-readable name of the dataset.
        Returns a response with internal id and name for a new dataset.
        :param payload: { "name": str }
        :return: new Dataset object
        """
        response = self._make_request({"name": name}, "dataset/create")
        return Dataset(response[DATASET_ID_KEY], self)

    def delete_dataset(self, dataset_id: str) -> dict:
        """
        Deletes a private dataset based on datasetId.
        Returns an empty payload where response status `200` indicates
        the dataset has been successfully deleted.
        :param payload: { "name": str }
        :return: { "dataset_id": str, "name": str }
        """
        return self._make_request({}, f"dataset/{dataset_id}", requests.delete)

    # TODO: maybe do more robust error handling here

    def populate_dataset(
        self,
        dataset_id: str,
        dataset_items: List[DatasetItem],
        batch_size: int = 20,
        force: bool = False
    ):
        """
        Appends images to a dataset with given dataset_id.
        Overwrites images on collision if forced.
        :param dataset_id: id of a dataset
        :param payload: { "items": List[DatasetItem], "force": bool }
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

        for item in dataset_items:
            local_items.append(item) if item.local else remote_items.append(item)

        local_batches = [
            local_items[i : i + batch_size] for i in range(0, len(local_items), batch_size)
        ]

        remote_batches = [
            remote_items[i: i + batch_size] for i in range(0, len(remote_items), batch_size)
        ]

        agg_response = UploadResponse(json={DATASET_ID_KEY: dataset_id})

        tqdm_local_batches = (
            tqdm.tqdm(local_batches)
        )

        tqdm_remote_batches = (
            tqdm.tqdm(remote_batches)
        )

        async_responses = []

        for batch in tqdm_local_batches:
            payload = construct_append_payload(batch)
            responses = self._process_append_requests_local(dataset_id, payload, batch_size, batch_size)
            async_responses.extend(responses)

        for batch in tqdm_remote_batches:
            payload = construct_append_payload(batch, force=force)
            responses = self._process_append_requests(dataset_id, payload, batch_size, batch_size)
            async_responses.extend(responses)

        for response in async_responses:
            agg_response.update_response(response.json())

        return agg_response

    def _process_append_requests_local(
        self,
        dataset_id: str,
        payload: dict,
        batch_size: int = 10,
        size: int = 10,
    ):
        def error() -> UploadResponse:
            return UploadResponse(
                {
                    DATASET_ID_KEY: dataset_id,
                    ERROR_ITEMS: 1,
                }
            )

        def exception_handler(request, exception):
            logger.error(request, exception)

        def preprocess_payload(item):
            image = open(item.get(IMAGE_URL_KEY), "rb")
            img_name = os.path.basename(image.name)
            img_type = f"image/{os.path.splitext(image.name)[1].strip('.')}"
            payload = {
                IMAGE_KEY: (img_name, image, img_type),
                ITEM_KEY: (None, json.dumps(item), "application/json"),
            }
            return payload

        items = payload[ITEMS_KEY]
        responses: List[Any] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            payloads = [preprocess_payload(item) for item in batch]

            async_requests = [
                self._make_grequest(
                    payload,
                    f"dataset/{dataset_id}/append",
                    local=True,
                )
                for payload in payloads
            ]

            async_responses = grequests.map(
                async_requests,
                exception_handler=exception_handler,
                size=size,
            )

            # don't forget to close all open files
            map(lambda x: x[IMAGE_KEY][1].close(), payloads)

            async_responses = [
                response if response.status_code == 200 else error()
                for response in async_responses
            ]
            responses.extend(async_responses)

        return responses

    def _process_append_requests(
        self,
        dataset_id: str,
        payload: dict,
        batch_size: int = 100,
        size: int = 10,
    ):
        def default_error(payload: dict) -> UploadResponse:
            return UploadResponse(
                {
                    DATASET_ID_KEY: dataset_id,
                    ERROR_ITEMS: len(payload[ITEMS_KEY]),
                }
            )

        def exception_handler(request, exception):
            logger.error(request, exception)
            return default_error(request.json())

        items = payload[ITEMS_KEY]
        payloads = [
            {ITEMS_KEY: items[i : i + batch_size]}
            for i in range(0, len(items), batch_size)
        ]

        async_requests = [
            self._make_grequest(
                payload,
                f"dataset/{dataset_id}/append",
                local=False,
            )
            for payload in payloads
        ]

        async_responses = grequests.map(
            async_requests, exception_handler=exception_handler, size=size
        )

        async_responses = [
            response if response.status_code == 200 else default_error(payload)
            for response, payload in zip(async_responses, payloads)
        ]

        return async_responses

    def annotate_dataset(self, dataset_id: str, annotations: List[DatasetItem], batch_size: int = 20):
        """
        Uploads ground truth annotations for a given dataset.
        :param dataset_id: id of the dataset
        :param annotations: List[DatasetItem]
        :return: {"dataset_id: str, "annotations_processed": int}
        """

        batches = [
            annotations[i: i + batch_size] for i in range(0, len(annotations), batch_size)
        ]

        agg_response = {DATASET_ID_KEY: dataset_id, ANNOTATIONS_PROCESSED_KEY: 0}

        tqdm_batches = (
            tqdm.tqdm(batches)
        )

        for batch in tqdm_batches:
            payload = construct_box_annotation_payload(batch)
            response = self._make_request(
                payload,
                f"dataset/{dataset_id}/annotate"
            )
            if STATUS_CODE_KEY in response:
                agg_response[ERRORS_KEY] = response
            else:
                agg_response[ANNOTATIONS_PROCESSED_KEY] += response[
                        ANNOTATIONS_PROCESSED_KEY
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
        return self._make_request(
            payload, f"dataset/{dataset_id}/ingest_tasks"
        )

    def add_model(self, name: str, reference_id: str, metadata: dict) -> Model:
        """
        Adds a model info to your repo based on payload params:
        name -- A human-readable name of the model project.
        reference_id -- An optional user-specified identifier to reference this given model.
        metadata -- An arbitrary metadata blob for the model.
        :param payload:
        {
            "name": str,
            "reference_id": str,
            "metadata": Dict[str, Any],
        }
        :return: { "model_id": str }
        """
        response = self._make_request(construct_model_creation_payload(name, reference_id, metadata), "models/add")
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
        response = self._make_request(
            payload, f"dataset/{dataset_id}/modelRun/create"
        )
        if response.get(STATUS_CODE_KEY, None):
            raise ModelRunCreationError(response.get("error"))

        return ModelRun(response[MODEL_RUN_ID_KEY], self)

    def predict(self, model_run_id: str, payload: dict):
        """
        Uploads model outputs as predictions for a model_run. Returns info about the upload.
        :param payload:
        {
            "annotations": List[Box2DPrediction],
        }
        :return:
        {
            "dataset_id": str,
            "model_run_id": str,
            "annotations_processed: int,
        }
        """
        return self._make_request(payload, f"modelRun/{model_run_id}/predict")

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
        return self._make_request(payload, f"modelRun/{model_run_id}/commit")

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
        return self._make_request(
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
        return self._make_request(
            {}, f"modelRun/{model_run_id}/info", requests.get
        )

    def dataitem_ref_id(self, dataset_id: str, reference_id: str):
        """
        :param dataset_id: internally controlled dataset id
        :param reference_id: reference_id of a dataset_item
        :return:
        """
        return self._make_request(
            {}, f"dataset/{dataset_id}/refloc/{reference_id}", requests.get
        )

    def predictions_ref_id(self, model_run_id: str, ref_id: str):
        """
        Returns Model Run info For Dataset Item by model_run_id and item reference_id.
        :param model_run_id: id of the model run.
        :param reference_id: reference_id of a dataset item.
        :return:
        {
            "annotations": List[Box2DPrediction],
        }
        """
        return self._make_request(
            {}, f"modelRun/{model_run_id}/refloc/{ref_id}", requests.get
        )

    def dataitem_iloc(self, dataset_id: str, i: int):
        """
        Returns Dataset Item info by dataset_id and absolute number of the dataset item.
        :param dataset_id:  internally controlled dataset id
        :param i: absolute number of the dataset_item
        :return:
        """
        return self._make_request(
            {}, f"dataset/{dataset_id}/iloc/{i}", requests.get
        )

    def predictions_iloc(self, model_run_id: str, i: int):
        """
        Returns Model Run Info For Dataset Item by model_run_id and absolute number of an item.
        :param model_run_id: id of the model run.
        :param i: absolute number of Dataset Item for a dataset corresponding to the model run.
        :return:
        {
            "annotations": List[Box2DPrediction],
        }
        """
        return self._make_request(
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
        return self._make_request(
            {}, f"dataset/{dataset_id}/loc/{dataset_item_id}", requests.get
        )

    def predictions_loc(self, model_run_id: str, dataset_item_id: str):
        """
        Returns Model Run Info For Dataset Item by its id.
        :param model_run_id: id of the model run.
        :param dataset_item_id: dataset_item_id of a dataset item.
        :return:
        {
            "annotations": List[Box2DPrediction],
        }
        """
        return self._make_request(
            {}, f"modelRun/{model_run_id}/loc/{dataset_item_id}", requests.get
        )

    def create_slice(self, dataset_id: str, payload: dict) -> Slice:
        """
        Creates a slice from items already present in a dataset.
        The caller must exclusively use either datasetItemIds or reference_ids
        as a means of identifying items in the dataset.

        "name" -- The human-readable name of the slice.

        "dataset_item_ids" -- An optional list of dataset item ids for the items in the slice

        "reference_ids" -- An optional list of user-specified identifier for the items in the slice

        :param
        dataset_id: id of the dataset
        payload:
        {
            "name": str,
            "dataset_item_ids": List[str],
            "reference_ids": List[str],
        }
        :return: new Slice object
        """
        response = self._make_request(
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

    def slice_info(
        self, slice_id: str, id_type: str = DATASET_ITEM_ID_KEY
    ) -> dict:
        """
        This endpoint provides information about specified slice.

        :param
        slice_id: id of the slice
        id_type: the type of IDs you want in response (either "reference_id" or "dataset_item_id")
        to identify the DatasetItems

        :return:
        {
            "name": str,
            "dataset_id": str,
            "dataset_item_ids": List[str],
        }
        """
        response = self._make_request(
            {},
            f"slice/{slice_id}?idType={id_type}",
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
        response = self._make_request(
            {},
            f"slice/{slice_id}",
            requests_command=requests.delete,
        )
        return response

    def _make_grequest(
        self,
        payload: dict,
        route: str,
        session=None,
        requests_command: Callable = grequests.post,
        local=True,
    ):
        """
        makes a grequest to Nucleus endpoint
        :param payload: file dict for multipart-formdata
        :param route: route for the request
        :param session: requests.session
        :param requests_command: grequests.post, grequests.get, grequests.delete
        :return: An async grequest object
        """

        endpoint = f"{NUCLEUS_ENDPOINT}/{route}"
        logger.info("Posting to %s", endpoint)

        if local:
            post = requests_command(
                endpoint,
                session=session,
                files=payload,
                auth=(self.api_key, ""),
            )
        else:
            post = requests_command(
                endpoint,
                session=session,
                json=payload,
                headers={"Content-Type": "application/json"},
                auth=(self.api_key, ""),
            )
        return post

    def _make_request(
        self, payload: dict, route: str, requests_command=requests.post
    ) -> dict:
        """
        makes a request to Nucleus endpoint
        :param payload: given payload
        :param route: route for the request
        :param requests_command: requests.post, requests.get, requests.delete
        :return: response
        """
        endpoint = f"{NUCLEUS_ENDPOINT}/{route}"
        logger.info("Posting to %s", endpoint)

        response = requests_command(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
        )
        logger.info("API request has response code %s", response.status_code)

        if response.status_code != 200:
            logger.warning(response)

        return (
            response.json()
        )  # TODO: this line fails if response has code == 404
