from .dataset import Dataset
from .prediction import BoxPrediction
from .model_run import ModelRun
from .payload_constructor import construct_box_predictions_payload
from typing import List

class Model:

    def __init__(self, model_id: str, name: str, reference_id: str, metadata: dict, client):
        self.id = model_id
        self.name = name
        self.reference_id = reference_id
        self.metadata = metadata
        self._client = client

    def create_run(self, name: str, metadata: dict, dataset: Dataset, predictions: List[BoxPrediction]) -> ModelRun:
        payload = {
            "name": name,
            "reference_id": self.reference_id,
            "metadata": metadata
        }
        model_run: ModelRun = self._client.create_model_run(dataset.id, payload)
        
        payload = construct_box_predictions_payload(predictions)

        model_run.predict(payload)

        return model_run