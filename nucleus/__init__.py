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

Box2DAnnotation:

item_id         |   str     |   The internally-controlled item identifier to associate this annotation with.
                |           |   The reference_id field should be empty if this field is populated.\n
reference_id    |   str     |   The user-specified reference identifier to associate this annotation with.\n
                |           |   The item_id field should be empty if this field is populated.
label	        |   str     |	The label for this annotation (e.g. car, pedestrian, bicycle).\n
type	        |   str     |   The type of this annotation. It should always be the box string literal.\n
x               |   float   |   The distance, in pixels, between the left border of the bounding box
                |           |   and the left border of the image.\n
y               |   float   |   The distance, in pixels, between the top border of the bounding box
                |           |   and the top border of the image.\n
width	        |   float   |   The width in pixels of the annotation.\n
height	        |   float   |   The height in pixels of the annotation.\n
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
x               |   float   |   The distance, in pixels, between the left border of the bounding box
                |           |   and the left border of the image.\n
y               |   float   |   The distance, in pixels, between the top border of the bounding box
                |           |   and the top border of the image.\n
width           |   float   |   The width in pixels of the annotation.\n
height          |   float   |   The height in pixels of the annotation.\n
metadata        |   dict    |   An arbitrary metadata blob for the annotation.\n
"""

import json
import logging
import os
from typing import List

import grequests
import requests

from .dataset import Dataset
from .upload_response import UploadResponse
from .model_run import ModelRun

logger = logging.getLogger(__name__)
logging.basicConfig()


NUCLEUS_ENDPOINT = "https://api.scale.com/v1/nucleus"
BATCH_SIZE = 10


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

    def list_datasets(self) -> List[str]:
        """
        Lists available datasets in your repo.
        :return: datasets_ids
        """
        # TODO implement API
        raise NotImplementedError

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

    def create_dataset(self, payload: dict) -> dict:
        """
        Creates a new dataset based on payload params:
        name -- A human-readable name of the dataset.
        Returns a response with internal id and name for a new dataset.
        :param payload: { "name": str }
        :return: { "dataset_id": str, "name": str }
        """
        return self._make_request(payload, "dataset/create")

    def populate_dataset(self, dataset_id: str, payload: dict, local=False):
        """
        Appends images to a dataset with given dataset_id. Overwrites images on collision if forced.
        :param dataset_id: id of a dataset
        :param payload: { "items": List[DatasetItem], "force": bool }
        :return:
        {
            "dataset_id: str,
            "new_items": int,
            "updated_items": int,
            "ignored_items": int,
        }
        """
        #TODO: maybe do more robust error handling here
        def exception_handler(request, exception):
            logger.error(exception)

        def format_payload_local(item):
            image = open(item.get("image_url"), "rb")
            img_name = os.path.basename(image.name)
            img_type = f"image/{os.path.splitext(image.name)[1].strip('.')}"

            files = {
                "image": (img_name, image, img_type),
                "item": (None, json.dumps(item), "application/json")
            }

            return files

        def process_responses(response_tuples: list, dataset_id: str):

            upload_response = UploadResponse(json={'dataset_id': dataset_id})

            for (response, num_uploads) in response_tuples:
                logger.info(response.status_code, response.json())
                if response and response.status_code == 200:
                    upload_response.update_response(response.json())
                else:
                    upload_response.record_error(response, num_uploads)

            return upload_response.as_dict()

        #TODO (Sasha): refactor to combine local_upload and batch_upload once we
        # implement batch local image upload for REST API
        def local_upload(dataset_id: str, payload: dict):
            async_requests = []
            num_uploads_per_request = []
            session = requests.session()
            items = payload.get("items", [])
            for item in items:
                payload = format_payload_local(item)
                async_requests.append(self._make_grequest(
                    payload, f"dataset/{dataset_id}/append", session=session, local = True))
            async_responses = grequests.map(
                async_requests, exception_handler=exception_handler)
            num_uploads_per_request.append(1) # temporary: to use in batching later
            return process_responses(zip(async_responses, num_uploads_per_request), dataset_id)

        def batch_upload(dataset_id: str, payload: dict):
            items = payload.get("items", [])
            num_uploads = len(items)
            async_requests = []
            num_uploads_per_request = []
            session = requests.session()
            num_batches = (num_uploads//BATCH_SIZE) + (num_uploads % BATCH_SIZE != 0)
            for i in range(num_batches):
                start_index = i*BATCH_SIZE
                end_index = min(len(items), (i+1)*BATCH_SIZE)
                curr_batch = items[start_index:end_index]
                payload = {"items": curr_batch}
                request = self._make_grequest(
                    payload, f"dataset/{dataset_id}/append", session=session, local = False)
                async_requests.append(request)
                num_uploads_per_request.append(end_index-start_index)

            async_responses = grequests.map(
                async_requests, exception_handler=exception_handler)

            return process_responses(zip(async_responses, num_uploads_per_request), dataset_id)

        return local_upload(dataset_id, payload) if local else batch_upload(dataset_id, payload)


    def annotate_dataset(self, dataset_id: str, payload: dict):
        # TODO batching logic if payload is too large
        """
        Uploads ground truth annotations for a given dataset.
        :param payload: {"annotations" : List[Box2DAnnotation]}
        :return: {"dataset_id: str, "annotations_processed": int}
        """
        return self._make_request(payload, f"dataset/{dataset_id}/annotate")

    def add_model(self, payload: dict) -> dict:
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
        return self._make_request(payload, "models/add")

    def create_model_run(self, dataset_id: str, payload: dict):
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
        :return:
        {
          "model_id": str,
          "model_run_id": str,
        }
        """
        return self._make_request(payload, f"dataset/{dataset_id}/modelRun/create")

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

    def commit_model_run(self, model_run_id: str):
        """
        Commits the model run.
        :return: {"model_run_id": str}
        """
        return self._make_request({}, f"modelRun/{model_run_id}/commit")

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
        return self._make_request({}, f"dataset/{dataset_id}/info", requests.get)

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
        return self._make_request({}, f"modelRun/{model_run_id}/info", requests.get)

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
        return self._make_request({}, f"dataset/{dataset_id}/iloc/{i}", requests.get)

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
        return self._make_request({}, f"dataset/{dataset_id}/loc/{dataset_item_id}", requests.get)

    def predictions_loc(self, model_run_id: str, dataset_item_id: str):
        """
        Returns Model Run Info For Dataset Item by its id.
        :param model_run_id: id of the model run.
        :param reference_id: reference_id of a dataset item.
        :return:
        {
            "annotations": List[Box2DPrediction],
        }
        """
        return self._make_request(
            {}, f"modelRun/{model_run_id}/loc/{dataset_item_id}", requests.get
        )

    def _make_grequest(self, payload: dict, route: str, session: requests.session, local = True):
        """
        makes a grequest to Nucleus endpoint
        :param files: file dict for multipart-formdata
        :param route: route for the request
        :param requests_command: requests.post, requests.get, requests.delete
        :return: An async grequest object
        """

        endpoint = f"{NUCLEUS_ENDPOINT}/{route}"
        logger.info('Posting to %s', endpoint)

        if local:
            post = grequests.post(
                endpoint,
                session=session,
                files=payload,
                auth=(self.api_key, ""),
            )
        else:
            post = grequests.post(
                endpoint,
                session=session,
                json=payload,
                headers = {"Content-Type": "application/json"},
                auth=(self.api_key, ""),
            )
        return post

    def _make_request(self, payload: dict, route: str, requests_command=requests.post) -> dict:
        """
        makes a request to Nucleus endpoint
        :param payload: given payload
        :param route: route for the request
        :param requests_command: requests.post, requests.get, requests.delete
        :return: response
        """
        endpoint = f"{NUCLEUS_ENDPOINT}/{route}"
        logger.info('Posting to %s', endpoint)

        response = requests_command(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            auth=(self.api_key, ""),
        )
        logger.info('API request has response code %s', response.status_code)

        if response.status_code != 200:
            logger.warning(response)

        return response.json() #TODO: this line fails if response has code == 404
