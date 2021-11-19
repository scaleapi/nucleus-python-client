"""
By uploading model predictions to Nucleus, you can compare your predictions to ground truth annotations and discover problems with your Models or Dataset.

You can also upload predictions for unannotated images, letting you query them based on model predictions. This can help you prioritize which unlabeled data to label next.

Within Nucleus, Models work in the following way:

1. You first create a Model. You can do this just once and reuse the model on multiple datasets.
2. You then upload predictions to a dataset.
3. Trigger calculation of model metrics in order to view model debugging insights.

Doing the three steps above allows you to visualize model performance within Nucleus, or compare multiple models that have been run on the same Dataset.


Note that you can always add more predictions to a dataset, but then you will need to re-run the calculation of metrics in order to have them be correct.

::

    import nucleus

    client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
    dataset = client.get_dataset("YOUR_DATASET_ID")
    prediction_1 = nucleus.BoxPrediction(
        label="label",
        x=0,
        y=0,
        width=10,
        height=10,
        reference_id="1",
        confidence=0.9,
        class_pdf={"label": 0.9, "other_label": 0.1},
    )
    prediction_2 = nucleus.BoxPrediction(
        label="label",
        x=0,
        y=0,
        width=10,
        height=10,
        reference_id="2",
        confidence=0.2,
        class_pdf={"label": 0.2, "other_label": 0.8},
    )
    model = client.add_model(
        name="My Model", reference_id="My-CNN", metadata={"timestamp": "121012401"}
    )
    # For small ingestions, we recommend synchronous ingestion
    response = dataset.upload_predictions(model, [prediction_1, prediction_2])
    # For large ingestions, we recommend asynchronous ingestion
    job = dataset.upload_predictions(
        [prediction_1, prediction_2], asynchronous=True
    )
    # Check current status
    job.status()
    # Sleep until ingestion is done
    job.sleep_until_complete()
    # Check errors
    job.errors()

    dataset.calculate_evaluation_metrics(model)
"""
from typing import List, Optional, Dict, Union

from nucleus.job import AsyncJob
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
    """A model that can be used to upload predictions to a dataset.

    Attributes:
        model_id: The scale-generated unique id for this model
        name: A human-readable name for the model
        reference_id: This is a unique, user-controlled ID for the model. This can be
            used, for example, to link to an external storage of models which may
            have its own id scheme.
        metadata: An arbitrary dictionary of additional data about this model that
            can be stored and retrieved. For example, you can store information
            about the hyperparameters used in training this model.
    """

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
        """Note: this method, as well as model runs in general are now deprecated.

        Instead models will automatically generate a model run when applied to a dataset
        using dataset.upload_predictions(model, predictions). Therefore there is no
        longer any need to create a model run, since you can upload predictions
        without needing to explicitly create a model run.

        When uploading to a dataset twice using the same model, the same model run
        will be reused by Nucleus.
        """
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

    def evaluate(self, unit_test_names: List[str]) -> AsyncJob:
        """Evaluates this on the specified Unit Tests. ::

            import nucleus.modelci as nm
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            model.evaluate(["sample_unit_test"])

        Args:
            model_id: ID of model to evaluate
            unit_test_names: list of unit tests to evaluate

        Returns:
            AsyncJob object of evaluation job
        """
        return self._client.modelci.evaluate_model_on_unit_tests(
            self.id, unit_test_names
        )
