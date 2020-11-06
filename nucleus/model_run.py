from typing import Optional


class ModelRun:
    """
    Model runs represent detections of a specific model on your dataset.
    Having an open model run is a prerequisite for uploading predictions to your dataset.
    """

    def __init__(self, model_run_id: str, client):
        self.model_run_id = model_run_id
        self._client = client

    @property
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

    def predict(self, payload: dict) -> dict:
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
        return self._client.predict(self.model_run_id, payload)

    def iloc(self, i: int) -> dict:
        """
        Returns Model Run Info For Dataset Item by its number.
        :param i: absolute number of Dataset Item for a dataset corresponding to the model run.
        :return:
        {
            "annotations": List[Box2DPrediction],
        }
        """
        return self._client.predictions_iloc(self.model_run_id, i)

    def refloc(self, reference_id: str) -> dict:
        """
        Returns Model Run Info For Dataset Item by its reference_id.
        :param reference_id: reference_id of a dataset item.
        :return:
        {
            "annotations": List[Box2DPrediction],
        }
        """
        return self._client.predictions_ref_id(self.model_run_id, reference_id)

    def loc(self, dataset_item_id: str) -> dict:
        """
        Returns Model Run Info For Dataset Item by its id.
        :param dataset_item_id: internally controlled id for dataset item.
        :return:
        {
            "annotations": List[Box2DPrediction],
        }
        """
        return self._client.predictions_loc(self.model_run_id, dataset_item_id)
