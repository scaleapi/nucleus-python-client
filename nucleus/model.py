from typing import List, Optional, Dict, Union
from .dataset import Dataset
from .prediction import (
    BoxPrediction,
    CuboidPrediction,
    PolygonPrediction,
    SegmentationPrediction,
)
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

    def __repr__(self):
        return f"Model(model_id='{self.id}', name='{self.name}', reference_id='{self.reference_id}', metadata={self.metadata}, client={self._client})"

    def __eq__(self, other):
        return (
            (self.id == other.id)
            and (self.name == other.name)
            and (self.metadata == other.metadata)
            and (self._client == other._client)
        )

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def from_json(cls, payload: dict, client):
        return cls(
            model_id=payload["id"],
            name=payload["name"],
            reference_id=payload["ref_id"],
            metadata=payload["metadata"] or None,
            client=client,
        )

    def create_run(
        self,
        name: str,
        dataset: Dataset,
        predictions: List[
            Union[
                BoxPrediction,
                PolygonPrediction,
                CuboidPrediction,
                SegmentationPrediction,
            ]
        ],
        metadata: Optional[Dict] = None,
        asynchronous: bool = False,
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

        model_run.predict(predictions, asynchronous=asynchronous)

        return model_run
