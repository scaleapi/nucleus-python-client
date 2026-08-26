from typing import Any, Dict, List, Optional, Union

import requests

from nucleus.annotation import check_all_mask_paths_remote
from nucleus.annotation_uploader import PredictionUploader
from nucleus.utils import (
    format_prediction_response,
    serialize_and_write_to_presigned_url,
)

from .async_job import AsyncJob
from .constants import (
    METADATA_KEY,
    MODEL_RUN_ID_KEY,
    MODEL_TAGS_KEY,
    MODEL_TRAINED_SLICE_IDS_KEY,
    NAME_KEY,
    REFERENCE_ID_KEY,
    REQUEST_ID_KEY,
    UPDATE_KEY,
)
from .dataset import Dataset
from .model_run import ModelRun
from .model_weights import ModelWeights
from .prediction import (
    BoxPrediction,
    CuboidPrediction,
    PolygonPrediction,
    Prediction,
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
    ):
        self.id = model_id
        self.name = name
        self.reference_id = reference_id
        self.metadata = metadata
        self.bundle_name = bundle_name
        self.tags = tags if tags else []
        self._client = client
        self.trained_slice_ids = trained_slice_ids if trained_slice_ids else []

    def __repr__(self):
        return f"Model(model_id='{self.id}', name='{self.name}', reference_id='{self.reference_id}', metadata={self.metadata}, bundle_name={self.bundle_name}, tags={self.tags}, client={self._client}, trained_slice_ids={self.trained_slice_ids})"

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
            reference_id=payload["ref_id"],
            metadata=payload["metadata"] or None,
            client=client,
            tags=payload.get(MODEL_TAGS_KEY, None),
            trained_slice_ids=payload.get(MODEL_TRAINED_SLICE_IDS_KEY, None),
        )

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

    def upload_predictions(
        self,
        predictions: List[Prediction],
        update: bool = False,
        asynchronous: bool = False,
        batch_size: int = 5000,
        remote_files_per_upload_request: int = 20,
        local_files_per_upload_request: int = 10,
    ) -> Union[Dict[str, Any], AsyncJob]:
        """Uploads predictions directly to this model, with no model run.

        This is the run-free ("model v2") prediction path: predictions are tied
        to the model itself as ``(model, dataset_item) -> prediction`` and are
        upserted server-side. Each prediction identifies its target item by
        ``dataset_item_id`` (the ``di_*`` id returned on exported items) or by
        ``reference_id``, so a single model can hold predictions for items that
        live in different datasets — no :class:`Dataset` or :class:`ModelRun` is
        needed. Reads go through :meth:`predictions_loc`,
        :meth:`predictions_refloc`, and :meth:`predictions_iloc`.

        Only ``box``, ``polygon``, and ``cuboid`` predictions are accepted on
        this path.

        The legacy run-based path (:meth:`Dataset.upload_predictions` /
        :meth:`ModelRun.add_predictions`) continues to work unchanged.

        Args:
            predictions: List of prediction objects to upload.
            update: If True, existing predictions for the same
                (reference_id, annotation_id) are overwritten. If False, they
                are skipped. Default is False.
            asynchronous: Whether or not to process the upload asynchronously
                (and return an :class:`AsyncJob` object). Default is False.
            batch_size: Number of predictions processed in each concurrent
                batch. Default is 5000. If you get timeouts when uploading
                geometric predictions, you can try lowering this batch size.
                This is only relevant for asynchronous=False.
            remote_files_per_upload_request: Number of remote files to upload in
                each request. Only relevant for asynchronous=False.
            local_files_per_upload_request: Number of local files to upload in
                each request. The maximum is 10. Only relevant for
                asynchronous=False.

        Returns:
            Payload describing the synchronous upload, or an :class:`AsyncJob`
            when ``asynchronous=True``::

                {
                    "model_id": str,
                    "predictions_processed": int,
                    "predictions_ignored": int,
                }
        """
        uploader = PredictionUploader(
            client=self._client,
            route=f"model/{self.id}/predictions",
        )
        uploader.check_for_duplicate_ids(predictions)

        if asynchronous:
            check_all_mask_paths_remote(predictions)
            request_id = serialize_and_write_to_presigned_url(
                predictions,
                dataset_id=None,
                client=self._client,
                route_prefix=f"model/{self.id}",
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"model/{self.id}/predictions?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        return uploader.upload(
            annotations=predictions,
            batch_size=batch_size,
            update=update,
            remote_files_per_upload_request=remote_files_per_upload_request,
            local_files_per_upload_request=local_files_per_upload_request,
        )

    def predictions_loc(self, dataset_item_id: str):
        """Fetches all of this model's predictions for a dataset item by its id.

        Model-scoped counterpart of :meth:`Dataset.prediction_loc` for the
        run-free prediction path.

        Parameters:
            dataset_item_id: Internally controlled id for the dataset item
                (``di_*``).

        Returns:
            Dictionary mapping prediction type to a list of prediction objects
            for this model::

                {
                    "box": List[BoxPrediction],
                    "polygon": List[PolygonPrediction],
                    "cuboid": List[CuboidPrediction],
                }
        """
        return format_prediction_response(
            self._client.make_request(
                payload=None,
                route=f"model/{self.id}/predictions/loc/{dataset_item_id}",
                requests_command=requests.get,
            )
        )

    def predictions_refloc(self, reference_id: str):
        """Fetches all of this model's predictions for a dataset item by its reference id.

        Model-scoped counterpart of :meth:`Dataset.predictions_refloc` for the
        run-free prediction path.

        Parameters:
            reference_id: User-defined reference id of the dataset item.

        Returns:
            Dictionary mapping prediction type to a list of prediction objects
            for this model::

                {
                    "box": List[BoxPrediction],
                    "polygon": List[PolygonPrediction],
                    "cuboid": List[CuboidPrediction],
                }
        """
        return format_prediction_response(
            self._client.make_request(
                payload=None,
                route=f"model/{self.id}/predictions/refloc/{reference_id}",
                requests_command=requests.get,
            )
        )

    def predictions_iloc(self, i: int):
        """Fetches all of this model's predictions for a dataset item by its index.

        Model-scoped counterpart of :meth:`Dataset.predictions_iloc` for the
        run-free prediction path.

        Parameters:
            i: Absolute index of the dataset item.

        Returns:
            Dictionary mapping prediction type to a list of prediction objects
            for this model::

                {
                    "box": List[BoxPrediction],
                    "polygon": List[PolygonPrediction],
                    "cuboid": List[CuboidPrediction],
                }
        """
        return format_prediction_response(
            self._client.make_request(
                payload=None,
                route=f"model/{self.id}/predictions/iloc/{i}",
                requests_command=requests.get,
            )
        )

    def copy_predictions_from_run(
        self, model_run_id: str, asynchronous: bool = True
    ) -> AsyncJob:
        """Copies predictions from a legacy v1 model run onto this model.

        Backfills the run-free ("model v2") prediction store for this model from
        an existing :class:`ModelRun`, so predictions previously uploaded via the
        run-based path become readable through :meth:`predictions_loc` and
        friends. The source run is left untouched.

        Args:
            model_run_id: Source model run id (``run_*``) to copy predictions
                from.
            asynchronous: Retained for forward compatibility; the copy always
                runs server-side as an async job. Default is True.

        Returns:
            An :class:`AsyncJob` tracking the copy.
        """
        response = self._client.make_request(
            {MODEL_RUN_ID_KEY: model_run_id},
            route=f"model/{self.id}/predictions/copyFromRun",
            requests_command=requests.post,
        )
        return AsyncJob.from_json(response, self._client)

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
