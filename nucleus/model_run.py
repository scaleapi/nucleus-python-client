from typing import Dict, List, Optional, Type, Union
import requests
from nucleus.annotation import check_all_mask_paths_remote
from nucleus.job import AsyncJob
from nucleus.utils import serialize_and_write_to_presigned_url

from .constants import (
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CUBOID_TYPE,
    DEFAULT_ANNOTATION_UPDATE_MODE,
    POLYGON_TYPE,
    REQUEST_ID_KEY,
    SEGMENTATION_TYPE,
    UPDATE_KEY,
)
from .prediction import (
    BoxPrediction,
    CuboidPrediction,
    PolygonPrediction,
    SegmentationPrediction,
)


class ModelRun:
    """
    Model runs represent detections of a specific model on your dataset.
    Having an open model run is a prerequisite for uploading predictions to your dataset.
    """

    def __init__(self, model_run_id: str, dataset_id: str, client):
        self.model_run_id = model_run_id
        self._client = client
        self._dataset_id = dataset_id

    def __repr__(self):
        return f"ModelRun(model_run_id='{self.model_run_id}', dataset_id='{self._dataset_id}', client={self._client})"

    def __eq__(self, other):
        if self.model_run_id == other.model_run_id:
            if self._client == other._client:
                return True
        return False

    def info(self) -> dict:
        """
        provides information about the Model Run:
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
        return self._client.model_run_info(self.model_run_id)

    def commit(self, payload: Optional[dict] = None) -> dict:
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
        return self._client.commit_model_run(self.model_run_id, payload)

    def predict(
        self,
        annotations: List[
            Union[
                BoxPrediction,
                PolygonPrediction,
                CuboidPrediction,
                SegmentationPrediction,
            ]
        ],
        update: Optional[bool] = DEFAULT_ANNOTATION_UPDATE_MODE,
        asynchronous: bool = False,
    ) -> Union[dict, AsyncJob]:
        """
        Uploads model outputs as predictions for a model_run. Returns info about the upload.
        :param annotations: List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        :return:
        {
            "model_run_id": str,
            "predictions_processed": int,
            "predictions_ignored": int,
        }
        """
        if asynchronous:
            check_all_mask_paths_remote(annotations)

            request_id = serialize_and_write_to_presigned_url(
                annotations, self._dataset_id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"modelRun/{self.model_run_id}/predict?async=1",
            )
            return AsyncJob.from_json(response, self._client)
        else:
            return self._client.predict(self.model_run_id, annotations, update)

    def iloc(self, i: int):
        """
        Returns Model Run Info For Dataset Item by its number.
        :param i: absolute number of Dataset Item for a dataset corresponding to the model run.
        :return: List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        }
        """
        response = self._client.predictions_iloc(self.model_run_id, i)
        return self._format_prediction_response(response)

    def refloc(self, reference_id: str):
        """
        Returns Model Run Info For Dataset Item by its reference_id.
        :param reference_id: reference_id of a dataset item.
        :return: List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        """
        response = self._client.predictions_ref_id(
            self.model_run_id, reference_id
        )
        return self._format_prediction_response(response)

    def loc(self, dataset_item_id: str):
        """
        Returns Model Run Info For Dataset Item by its id.
        :param dataset_item_id: internally controlled id for dataset item.
        :return:
        {
            "annotations": List[Box2DPrediction],
        }
        """
        response = self._client.predictions_loc(
            self.model_run_id, dataset_item_id
        )
        return self._format_prediction_response(response)

    def ungrouped_export(self):
        json_response = self._client.make_request(
            payload={},
            route=f"modelRun/{self.model_run_id}/ungrouped",
            requests_command=requests.get,
        )
        return self._format_prediction_response(
            {ANNOTATIONS_KEY: json_response}
        )

    def _format_prediction_response(
        self, response: dict
    ) -> Union[
        dict,
        List[
            Union[
                BoxPrediction,
                PolygonPrediction,
                CuboidPrediction,
                SegmentationPrediction,
            ]
        ],
    ]:
        annotation_payload = response.get(ANNOTATIONS_KEY, None)
        if not annotation_payload:
            # An error occurred
            return response
        annotation_response = {}
        type_key_to_class: Dict[
            str,
            Union[
                Type[BoxPrediction],
                Type[PolygonPrediction],
                Type[CuboidPrediction],
                Type[SegmentationPrediction],
            ],
        ] = {
            BOX_TYPE: BoxPrediction,
            POLYGON_TYPE: PolygonPrediction,
            CUBOID_TYPE: CuboidPrediction,
            SEGMENTATION_TYPE: SegmentationPrediction,
        }
        for type_key in annotation_payload:
            type_class = type_key_to_class[type_key]
            annotation_response[type_key] = [
                type_class.from_json(annotation)
                for annotation in annotation_payload[type_key]
            ]
        return annotation_response
