from typing import List, Optional, Dict, Union
from .dataset import Dataset
from .prediction import BoxPrediction, PolygonPrediction
from .model_run import ModelRun
from .constants import (
    NAME_KEY,
    REFERENCE_ID_KEY,
    METADATA_KEY,
)


class Model:
    def __init__(
        self,
        model_id: str,
        name: str,
        reference_id: str,
        metadata: Optional[Dict],
        client,
    ):
        self.id = model_id
        self.name = name
        self.reference_id = reference_id
        self.metadata = metadata
        self._client = client

    def create_run(
        self,
        name: str,
        dataset: Dataset,
        predictions: List[
            Union[BoxPrediction, PolygonPrediction, SegmentationPrediction]
        ],
        metadata: Optional[Dict] = None,
    ) -> ModelRun:
        payload: dict = {
            NAME_KEY: name,
            REFERENCE_ID_KEY: self.reference_id,
        }
        if metadata:
            payload[METADATA_KEY] = metadata
        model_run: ModelRun = self._client.create_model_run(
            dataset.id, payload
        )

        model_run.predict(predictions)

        return model_run
