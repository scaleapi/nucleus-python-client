from typing import Dict, List, Optional, Union

import requests

from .async_job import AsyncJob
from .constants import (
    BACKEND_REFERENCE_ID_KEY,
    METADATA_KEY,
    MODEL_ARCHITECTURE_KEY,
    MODEL_BUMP_TYPE_KEY,
    MODEL_BUNDLE_NAME_KEY,
    MODEL_DESCRIPTION_KEY,
    MODEL_INPUT_SCHEMA_KEY,
    MODEL_NUM_PARAMETERS_KEY,
    MODEL_OUTPUT_SCHEMA_KEY,
    MODEL_PARENT_MODEL_PROJECT_ID_KEY,
    MODEL_TAGS_KEY,
    MODEL_TRAINED_SLICE_IDS_KEY,
    MODEL_TRAINING_DATA_KEY,
    MODEL_VERSION_LABEL_KEY,
    MODEL_VERSION_MAJOR_KEY,
    MODEL_VERSION_MINOR_KEY,
    NAME_KEY,
    REFERENCE_ID_KEY,
)
from .dataset import Dataset
from .model_run import ModelRun
from .model_weights import ModelWeights
from .payload_constructor import NO_UPDATE
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

    1. You first :meth:`create a Model<NucleusClient.create_model>`. You can do this
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

        model = client.create_model(
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
    endpoint, using :meth:`NucleusClient.create_model`.
    """

    def __init__(
        self,
        model_id,
        name,
        reference_id,
        metadata,
        client,
        bundle_name=None,
        tags=None,
        trained_slice_ids=None,
        description=None,
        architecture=None,
        num_parameters=None,
        training_data=None,
        input_schema=None,
        output_schema=None,
        parent_model_project_id=None,
        version_major=None,
        version_minor=None,
        version_label=None,
    ):
        self.id = model_id
        self.name = name
        self.reference_id = reference_id
        self.metadata = metadata
        self.bundle_name = bundle_name
        self.tags = tags if tags else []
        self._client = client
        self.trained_slice_ids = trained_slice_ids if trained_slice_ids else []
        self.description = description
        self.architecture = architecture
        self.num_parameters = num_parameters
        self.training_data = training_data
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.parent_model_project_id = parent_model_project_id
        self.version_major = version_major
        self.version_minor = version_minor
        self.version_label = version_label

    def __repr__(self):
        return (
            f"Model(model_id='{self.id}', name='{self.name}', "
            f"reference_id='{self.reference_id}', metadata={self.metadata}, "
            f"bundle_name={self.bundle_name}, tags={self.tags}, "
            f"trained_slice_ids={self.trained_slice_ids}, "
            f"description={self.description}, architecture={self.architecture}, "
            f"num_parameters={self.num_parameters}, training_data={self.training_data}, "
            f"input_schema={self.input_schema}, output_schema={self.output_schema}, "
            f"parent_model_project_id={self.parent_model_project_id}, "
            f"version_major={self.version_major}, version_minor={self.version_minor}, "
            f"version_label={self.version_label}, client={self._client})"
        )

    def __eq__(self, other):
        return (
            (self.id == other.id)
            and (self.name == other.name)
            and (self.metadata == other.metadata)
            and (self._client == other._client)
            and (self.bundle_name == other.bundle_name)
            and (self.trained_slice_ids == other.trained_slice_ids)
        )

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def from_json(cls, payload: dict, client):
        """Instantiates model object from schematized JSON dict payload."""
        return cls(
            model_id=payload["id"],
            name=payload["name"],
            # The backend returns the reference id as `ref_id`; fall back to the
            # canonical key for any endpoint that returns it that way.
            reference_id=payload.get(
                BACKEND_REFERENCE_ID_KEY, payload.get(REFERENCE_ID_KEY)
            ),
            metadata=payload["metadata"] or None,
            client=client,
            bundle_name=payload.get(MODEL_BUNDLE_NAME_KEY, None),
            tags=payload.get(MODEL_TAGS_KEY, None),
            trained_slice_ids=payload.get(MODEL_TRAINED_SLICE_IDS_KEY, None),
            description=payload.get(MODEL_DESCRIPTION_KEY, None),
            architecture=payload.get(MODEL_ARCHITECTURE_KEY, None),
            num_parameters=payload.get(MODEL_NUM_PARAMETERS_KEY, None),
            training_data=payload.get(MODEL_TRAINING_DATA_KEY, None),
            input_schema=payload.get(MODEL_INPUT_SCHEMA_KEY, None),
            output_schema=payload.get(MODEL_OUTPUT_SCHEMA_KEY, None),
            parent_model_project_id=payload.get(
                MODEL_PARENT_MODEL_PROJECT_ID_KEY, None
            ),
            version_major=payload.get(MODEL_VERSION_MAJOR_KEY, None),
            version_minor=payload.get(MODEL_VERSION_MINOR_KEY, None),
            version_label=payload.get(MODEL_VERSION_LABEL_KEY, None),
        )

    def update(
        self,
        name=NO_UPDATE,
        reference_id=NO_UPDATE,
        metadata=NO_UPDATE,
        description=NO_UPDATE,
        architecture=NO_UPDATE,
        num_parameters=NO_UPDATE,
        training_data=NO_UPDATE,
        input_schema=NO_UPDATE,
        output_schema=NO_UPDATE,
        training_data_fields=NO_UPDATE,
    ) -> "Model":
        """Edit this model's descriptive fields in place.

        Thin wrapper over :meth:`NucleusClient.update_model` — only the fields you pass
        are changed, and passing ``None`` clears a nullable field. Refreshes this
        object's attributes from the server response and returns ``self``.
        """
        updated = self._client.update_model(
            self.id,
            name=name,
            reference_id=reference_id,
            metadata=metadata,
            description=description,
            architecture=architecture,
            num_parameters=num_parameters,
            training_data=training_data,
            input_schema=input_schema,
            output_schema=output_schema,
            training_data_fields=training_data_fields,
        )
        # Copy refreshed state back onto this instance (both share the same client).
        self.__dict__.update(updated.__dict__)
        return self

    def set_parent(
        self,
        parent_model_project_id: Optional[str],
        bump_type: Optional[str] = None,
        version_major: Optional[int] = None,
        version_minor: Optional[int] = None,
        version_label: Optional[str] = None,
    ) -> "Model":
        """Set or clear this model's parent and version.

        Pass a model id to make this model a version branched from that parent, or
        ``None`` to clear the parent and turn it back into a root. ``bump_type``
        (``"major"``/``"minor"``) selects the version bump relative to the parent unless
        ``version_major``/``version_minor`` are given explicitly. Updates this object's
        version fields from the server response and returns ``self``.
        """
        payload: Dict[str, Union[str, int, None]] = {
            MODEL_PARENT_MODEL_PROJECT_ID_KEY: parent_model_project_id,
        }
        if bump_type is not None:
            payload[MODEL_BUMP_TYPE_KEY] = bump_type
        if version_major is not None:
            payload[MODEL_VERSION_MAJOR_KEY] = version_major
        if version_minor is not None:
            payload[MODEL_VERSION_MINOR_KEY] = version_minor
        if version_label is not None:
            payload[MODEL_VERSION_LABEL_KEY] = version_label

        response = self._client.make_request(
            payload, f"model/{self.id}/parent"
        )
        self.parent_model_project_id = response.get(
            MODEL_PARENT_MODEL_PROJECT_ID_KEY, self.parent_model_project_id
        )
        self.version_major = response.get(
            MODEL_VERSION_MAJOR_KEY, self.version_major
        )
        self.version_minor = response.get(
            MODEL_VERSION_MINOR_KEY, self.version_minor
        )
        self.version_label = response.get(
            MODEL_VERSION_LABEL_KEY, self.version_label
        )
        return self

    def create_run(
        self,
        name: str,
        dataset: Optional[Dataset] = None,
        predictions: Optional[
            List[
                Union[
                    BoxPrediction,
                    PolygonPrediction,
                    CuboidPrediction,
                    SegmentationPrediction,
                ]
            ]
        ] = None,
        metadata: Optional[Dict] = None,
        asynchronous: bool = False,
        reference_id: Optional[str] = None,
    ) -> ModelRun:
        """Creates a model run for this model.

        Call it with just a name to create a run, then attach predictions with
        :meth:`ModelRun.add_predictions`::

            run = model.create_run(name="my-run")
            run.add_predictions(predictions)

        or pass ``predictions`` directly to create and upload in one call::

            run = model.create_run(name="my-run", predictions=predictions)

        Each prediction identifies its target item by ``dataset_item_id`` (the
        ``di_*`` id returned on exported items), so predictions can come from
        anywhere and a single run can cover items across different datasets.

        Args:
            name: Human-readable name for the model run.
            predictions: Optional predictions to attach to the run immediately.
            metadata: Optional arbitrary metadata blob for the run.
            reference_id: Optional user-defined reference id for the run.
            dataset: Deprecated. Passing a dataset uses the legacy path that
                binds the run to that one dataset and uploads ``predictions``
                to it. Omit it to use the recommended flow above.
            asynchronous: Only used by the deprecated dataset-bound path.

        Returns:
            The created :class:`ModelRun`.
        """
        # Legacy path: an explicit dataset binds the run to that dataset and
        # uploads predictions to it. Kept for backwards compatibility; new code
        # should omit `dataset` and use `run.add_predictions(...)`.
        if dataset is not None:
            payload: dict = {
                NAME_KEY: name,
                REFERENCE_ID_KEY: self.reference_id,
            }
            if metadata:
                payload[METADATA_KEY] = metadata
            model_run: ModelRun = self._client.create_model_run(
                dataset.id, payload
            )
            model_run.predict(predictions or [], asynchronous=asynchronous)
            return model_run

        response = self._client.make_request(
            {
                NAME_KEY: name,
                REFERENCE_ID_KEY: reference_id,
                METADATA_KEY: metadata or {},
            },
            route=f"model/{self.id}/modelRun/create",
            requests_command=requests.post,
        )
        run = ModelRun(
            model_run_id=response["model_run_id"],
            dataset_id=None,
            client=self._client,
        )
        if predictions:
            run.add_predictions(predictions)
        return run

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

    def run(
        self, dataset_id: str, model_run_name: str, slice_id: Optional[str]
    ) -> str:
        """Runs inference on the bundle associated with the model on the dataset. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]

            model.run("ds_123456")

        Args:
            dataset_id: The ID of the dataset to run inference on.
            model_run_name: The name of the model run.
            slice_id: The ID of the slice of the dataset to run inference on.

        Returns:
            job_id: The ID of the :class:`AsyncJob` used to track job progress.
        """
        response = self._client.make_request(
            {
                "dataset_id": dataset_id,
                "slice_id": slice_id,
                "model_run_name": model_run_name,
            },
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
        response: requests.Response = self._client.make_request(
            {MODEL_TAGS_KEY: tags},
            f"model/{self.id}/tag",
            requests_command=requests.post,
            return_raw_response=True,
        )

        if response.ok:
            for tag in tags:
                if tag not in self.tags:
                    self.tags.append(tag)

        return response.json()

    def remove_tags(self, tags: List[str]):
        """Remove tag(s) from the model. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]

            model.remove_tags(["tag_x"])

        Args:
            tags: list of tag names to remove
        """
        response: requests.Response = self._client.make_request(
            {MODEL_TAGS_KEY: tags},
            f"model/{self.id}/tag",
            requests_command=requests.delete,
            return_raw_response=True,
        )

        if response.ok:
            self.tags = list(filter(lambda t: t not in tags, self.tags))

        return response.json()

    def add_trained_slice_ids(self, slice_ids: List[str]):
        """Add trained slice id(s) to the model. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]

            model.add_trained_slice_ids(["slc_...", "slc_..."])

        Args:
            slice_ids: list of trained slice ids
        """
        response: requests.Response = self._client.make_request(
            {MODEL_TRAINED_SLICE_IDS_KEY: slice_ids},
            f"model/{self.id}/trainedSliceId",
            requests_command=requests.post,
            return_raw_response=True,
        )

        if response.ok:
            for slice_id in slice_ids:
                if slice_id not in self.trained_slice_ids:
                    self.trained_slice_ids.append(slice_id)

        return response.json()

    def remove_trained_slice_ids(self, slide_ids: List[str]):
        """Remove trained slice id(s) from the model. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]

            model.remove_trained_slice_ids(["slc_...", "slc_..."])

        Args:
            slice_ids: list of trained slice ids to remove
        """
        response: requests.Response = self._client.make_request(
            {MODEL_TRAINED_SLICE_IDS_KEY: slide_ids},
            f"model/{self.id}/trainedSliceId",
            requests_command=requests.delete,
            return_raw_response=True,
        )

        if response.ok:
            self.trained_slice_ids = list(
                filter(lambda t: t not in slide_ids, self.trained_slice_ids)
            )

        return response.json()

    def upload_weights(
        self,
        path: str,
        *,
        content_type: Optional[str] = None,
        original_filename: Optional[str] = None,
        checksum_sha256: Optional[str] = None,
        progress: bool = True,
    ) -> ModelWeights:
        """Attach a weights artifact to this model. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.get_model(reference_id="My-CNN")

            model.upload_weights("/path/to/weights.bin")

        See :meth:`NucleusClient.upload_model_weights` for the accepted keyword
        arguments.
        """
        return self._client.upload_model_weights(
            self,
            path,
            content_type=content_type,
            original_filename=original_filename,
            checksum_sha256=checksum_sha256,
            progress=progress,
        )

    def download_weights(self, path: str, *, progress: bool = True) -> str:
        """Download this model's weights artifact to ``path``.

        See :meth:`NucleusClient.download_model_weights`.
        """
        return self._client.download_model_weights(
            self, path, progress=progress
        )

    def weights(self) -> ModelWeights:
        """Fetch metadata for this model's weights artifact.

        See :meth:`NucleusClient.get_model_weights`.
        """
        return self._client.get_model_weights(self)

    def delete_weights(self) -> bool:
        """Delete this model's weights artifact.

        See :meth:`NucleusClient.delete_model_weights`.
        """
        return self._client.delete_model_weights(self)
