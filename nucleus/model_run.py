"""
Model Runs are deprecated and will be removed in a future version of the python client.
It is now possible to upload model predictions without a need for creating a model run

For example::

    import nucleus
    client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
    prediction_1 = nucleus.BoxPrediction(label="label", x=0, y=0, width=10, height=10, reference_id="1", confidence=0.9, class_pdf={'label': 0.9, 'other_label': 0.1})
    prediction_2 = nucleus.BoxPrediction(label="label", x=0, y=0, width=10, height=10, reference_id="2", confidence=0.2, class_pdf={'label': 0.2, 'other_label': 0.8})
    model = client.add_model(name="My Model", reference_id="My-CNN", metadata={"timestamp": "121012401"})
    response = dataset.upload_predictions(model, [prediction_1, prediction_2])
"""


from typing import List, Optional, Union

import requests

from nucleus.annotation import check_all_mask_paths_remote
from nucleus.annotation_uploader import PredictionUploader
from nucleus.job import AsyncJob
from nucleus.utils import (
    format_prediction_response,
    serialize_and_write_to_presigned_url,
)

from .constants import (
    ANNOTATIONS_KEY,
    DEFAULT_ANNOTATION_UPDATE_MODE,
    REQUEST_ID_KEY,
    UPDATE_KEY,
)
from .prediction import (
    BoxPrediction,
    CuboidPrediction,
    PolygonPrediction,
    SegmentationPrediction,
    from_json,
)


class ModelRun:
    """
    This class is deprecated and will be removed from the python client.
    """

    def __init__(self, model_run_id: str, dataset_id: str, client):
        self.model_run_id = model_run_id
        self._client = client
        self.dataset_id = dataset_id

    def __repr__(self):
        return f"ModelRun(model_run_id='{self.model_run_id}', dataset_id='{self.dataset_id}', client={self._client})"

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
        update: bool = DEFAULT_ANNOTATION_UPDATE_MODE,
        asynchronous: bool = False,
        batch_size: int = 5000,
        remote_files_per_upload_request: int = 20,
        local_files_per_upload_request: int = 10,
        local_file_upload_concurrency: int = 30,
    ) -> Union[dict, AsyncJob]:
        """
        Uploads model outputs as predictions for a model_run. Returns info about the upload.

        Args:
            annotations: Predictions to upload for this model run,
            update: If True, existing predictions for the same (reference_id, annotation_id)
                will be overwritten. If False, existing predictions will be skipped.
            asynchronous: Whether or not to process the upload asynchronously (and
                return an :class:`AsyncJob` object). Default is False.
            batch_size: Number of predictions processed in each concurrent batch.
                Default is 5000. If you get timeouts when uploading geometric annotations,
                you can try lowering this batch size. This is only relevant for
                asynchronous=False.
            remote_files_per_upload_request: Number of remote files to upload in each
                request. Segmentations have either local or remote files, if you are
                getting timeouts while uploading segmentations with remote urls, you
                should lower this value from its default of 20. This is only relevant for
                asynchronous=False
            local_files_per_upload_request: Number of local files to upload in each
                request. Segmentations have either local or remote files, if you are
                getting timeouts while uploading segmentations with local files, you
                should lower this value from its default of 10. The maximum is 10.
                This is only relevant for asynchronous=False
            local_file_upload_concurrency: Number of concurrent local file uploads.
                This is only relevant for asynchronous=False
        :return:
        {
            "model_run_id": str,
            "predictions_processed": int,
            "predictions_ignored": int,
        }
        """
        uploader = PredictionUploader(
            model_run_id=self.model_run_id, client=self._client
        )
        uploader.check_for_duplicate_ids(annotations)

        if asynchronous:
            check_all_mask_paths_remote(annotations)

            request_id = serialize_and_write_to_presigned_url(
                annotations, self.dataset_id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"modelRun/{self.model_run_id}/predict?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        return uploader.upload(
            annotations=annotations,
            update=update,
            batch_size=batch_size,
            remote_files_per_upload_request=remote_files_per_upload_request,
            local_files_per_upload_request=local_files_per_upload_request,
            local_file_upload_concurrency=local_file_upload_concurrency,
        )

    def iloc(self, i: int):
        """
        Returns Model Run Info For Dataset Item by its number.
        :param i: absolute number of Dataset Item for a dataset corresponding to the model run.
        :return: List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        }
        """
        response = self._client.predictions_iloc(self.model_run_id, i)
        return format_prediction_response(response)

    def refloc(self, reference_id: str):
        """
        Returns Model Run Info For Dataset Item by its reference_id.
        :param reference_id: reference_id of a dataset item.
        :return: List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction, SegmentationPrediction]],
        """
        response = self._client.get(
            f"modelRun/{self.model_run_id}/refloc/{reference_id}"
        )
        return format_prediction_response(response)

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
        return format_prediction_response(response)

    def prediction_loc(self, reference_id: str, annotation_id: str):
        """
        Returns info for single Prediction by its reference id and annotation id.
        :param reference_id: the user specified id for the image
        :param annotation_id: the user specified id for the prediction, or if one was not provided, the Scale internally generated id for the prediction
        :return:
         BoxPrediction | PolygonPrediction | CuboidPrediction
        """

        response = self._client.make_request(
            {},
            f"modelRun/{self.model_run_id}/prediction/loc/{reference_id}/{annotation_id}",
            requests.get,
        )

        return from_json(response)

    def ungrouped_export(self):
        json_response = self._client.make_request(
            payload={},
            route=f"modelRun/{self.model_run_id}/ungrouped",
            requests_command=requests.get,
        )
        return format_prediction_response({ANNOTATIONS_KEY: json_response})
