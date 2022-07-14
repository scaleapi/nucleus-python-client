from typing import Dict, List, Optional, Union

import requests

from .constants import METADATA_KEY, MODEL_TAGS_KEY, NAME_KEY, REFERENCE_ID_KEY
from .dataset import Dataset
from .job import AsyncJob
from .model_run import ModelRun
from .prediction import (
    BoxPrediction,
    CuboidPrediction,
    PolygonPrediction,
    SegmentationPrediction,
)


class Model:
    """A model that can be used to upload predictions to a dataset.

    By uploading model predictions to Nucleus, you can compare your predictions
    to ground truth annotations and discover problems with your Models or
    :class:`Dataset`.

    You can also upload predictions for unannotated images, letting you query
    them based on model predictions. This can help you prioritize which
    unlabeled data to label next.

    Within Nucleus, Models work in the following way:

    1. You first :meth:`create a Model<NucleusClient.add_model>`. You can do this
       just once and reuse the model on multiple datasets.
    2. You then :meth:`upload predictions <Dataset.upload_predictions>` to a dataset.
    3. Trigger :meth:`calculation of metrics <Dataset.calculate_evaluation_metrics>`
       in order to view model debugging insights.

    The above steps above will allow you to visualize model performance within
    Nucleus, or compare multiple models that have been run on the same Dataset.

    Note that you can always add more predictions to a dataset, but then you
    will need to re-run the calculation of metrics in order to have them be
    correct.

    ::

        import nucleus

        client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
        dataset = client.get_dataset(YOUR_DATASET_ID)

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
            model, [prediction_1, prediction_2], asynchronous=True
        )
        # Check current status
        job.status()
        # Sleep until ingestion is done
        job.sleep_until_complete()
        # Check errors
        job.errors()

        dataset.calculate_evaluation_metrics(model)

    Models cannot be instantiated directly and instead must be created via API
    endpoint, using :meth:`NucleusClient.add_model`.
    """

    def __init__(
        self,
        model_id,
        name,
        reference_id,
        metadata,
        client,
        bundle_name=None,
        tags: List[str] = None,
    ):
        self.id = model_id
        self.name = name
        self.reference_id = reference_id
        self.metadata = metadata
        self.bundle_name = bundle_name
        self.tags = tags if tags else []
        self._client = client

    def __repr__(self):
        return f"Model(model_id='{self.id}', name='{self.name}', reference_id='{self.reference_id}', metadata={self.metadata}, bundle_name={self.bundle_name}, client={self._client})"

    def __eq__(self, other):
        return (
            (self.id == other.id)
            and (self.name == other.name)
            and (self.metadata == other.metadata)
            and (self._client == other._client)
            and (self.bundle_name == other.bundle_name)
        )

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def from_json(cls, payload: dict, client):
        """Instantiates model object from schematized JSON dict payload."""
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
        # This method, as well as model runs in general are now deprecated.

        # Instead models will automatically generate a model run when applied to
        # a dataset using dataset.upload_predictions(model, predictions). Therefore
        # there is no longer any need to create a model run, since you can upload
        # predictions without needing to explicitly create a model run.

        # When uploading to a dataset twice using the same model, the same model
        # run will be reused by Nucleus.

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

    def evaluate(self, scenario_test_names: List[str]) -> AsyncJob:
        """Evaluates this on the specified Unit Tests. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]
            scenario_test = client.validate.create_scenario_test(
                "sample_scenario_test", "YOUR_SLICE_ID"
            )

            model.evaluate(["sample_scenario_test"])

        Args:
            scenario_test_names: list of unit tests to evaluate

        Returns:
            AsyncJob object of evaluation job
        """
        response = self._client.make_request(
            {"test_names": scenario_test_names},
            f"validate/{self.id}/evaluate",
            requests_command=requests.post,
        )
        return AsyncJob.from_json(response, self._client)

    def run(self, dataset_id: str, slice_id: Optional[str]) -> str:
        """Runs inference on the bundle associated with the model on the dataset. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]

            model.run("ds_123456")

        Args:
            dataset_id: id of dataset to run inference on
            job_id: nucleus job used to track async job progress
            slice_id: (optional) id of slice of the dataset to run inference on
        """
        response = self._client.make_request(
            {"dataset_id": dataset_id, "slice_id": slice_id},
            f"model/run/{self.id}/",
            requests_command=requests.post,
        )

        return response

    def add_tags(self, tags: List[str]):
        """Tag the model with custom tag names. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]

            model.add_tags(["tag_A", "tag_B"])

        Args:
            tags: list of tag names
        """
        response = self._client.make_request(
            {MODEL_TAGS_KEY: tags},
            f"model/{self.id}/tag",
            requests_command=requests.post,
        )

        if response.get("msg", False):
            self.tags.extend(tags)

        return response

    def remove_tags(self, tags: List[str]):
        """Remove tag(s) from the model. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]

            model.remove_tags(["tag_x"])

        Args:
            tags: list of tag names to remove
        """
        response = self._client.make_request(
            {MODEL_TAGS_KEY: tags},
            f"model/{self.id}/tag",
            requests_command=requests.delete,
        )

        if response.get("msg", False):
            self.tags = list(filter(lambda t: t not in tags, self.tags))

        return response
