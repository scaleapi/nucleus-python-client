from typing import List, Dict, Any
from .dataset_item import DatasetItem
from .annotation import BoxAnnotation
from .constants import (
    DATASET_NAME_KEY,
    DATASET_MODEL_RUNS_KEY,
    DATASET_SLICES_KEY,
    DATASET_LENGTH_KEY,
    DATASET_ITEM_IDS_KEY,
    REFERENCE_IDS_KEY,
    NAME_KEY,
)


class Dataset:
    """
    Nucleus Dataset. You can append images with metadata to your dataset,
    annotate it with ground truth and upload model predictions to evaluate and
    compare model performance on you data.
    """

    def __init__(self, dataset_id: str, client):
        self.id = dataset_id
        self._client = client

    @property
    def name(self) -> str:
        return self.info().get(DATASET_NAME_KEY, "")

    @property
    def model_runs(self) -> List[str]:
        return self.info().get(DATASET_MODEL_RUNS_KEY, [])

    @property
    def slices(self) -> List[str]:
        return self.info().get(DATASET_SLICES_KEY, [])

    @property
    def size(self) -> int:
        return self.info().get(DATASET_LENGTH_KEY, 0)

    @property
    def items(self) -> List[DatasetItem]:
        return self._client.get_dataset_items(self.id)

    def info(self) -> dict:
        """
        Returns information about existing dataset
        :return: dictionary of the form
            {
                'name': str,
                'length': int,
                'model_run_ids': List[str],
                'slice_ids': List[str]
            }
        """
        return self._client.dataset_info(self.id)

    def create_model_run(self, payload: dict):
        """
        Creates model run for the dataset based on the given parameters:

        'reference_id' -- The user-specified reference identifier to associate with the model.
                        The 'model_id' field should be empty if this field is populated.

        'model_id' -- The internally-controlled identifier of the model.
                    The 'reference_id' field should be empty if this field is populated.

        'name' -- An optional name for the model run.

        'metadata' -- An arbitrary metadata blob for the current run.

        :param payload:
        {
            "reference_id": str,
            "model_id": str,
            "name": Optional[str],
            "metadata": Optional[Dict[str, Any]],
        }
        :return:
        {
          "model_id": str,
          "model_run_id": str,
        }
        """
        return self._client.create_model_run(self.id, payload)

    def annotate(
        self, annotations: List[BoxAnnotation], batch_size=20
    ) -> dict:
        """
        Uploads ground truth annotations for a given dataset.
        :param payload: {"annotations" : List[Box2DAnnotation]}
        :return:
        {
            "dataset_id: str,
            "new_items": int,
            "updated_items": int,
            "ignored_items": int,
        }
        """
        return self._client.annotate_dataset(
            self.id, annotations, batch_size=batch_size
        )

    def ingest_tasks(self, task_ids: dict):
        """
        If you already submitted tasks to Scale for annotation this endpoint ingests your completed tasks
        annotated by Scale into your Nucleus Dataset.
        Right now we support ingestion from Videobox Annotation and 2D Box Annotation projects.
        Lated we'll supoport more annotation types.
        :param payload: {"tasks" : List[task_ids]}
        :return: {"ingested_tasks": int, "ignored_tasks": int, "pending_tasks": int}
        """
        return self._client.ingest_tasks(self.id, {"tasks": task_ids})

    def append(self, dataset_items: List[DatasetItem], batch_size=20) -> dict:
        """
        Appends images with metadata (dataset items) to the dataset. Overwrites images on collision if forced.

        :param payload: {"items": List[DatasetItem], "force": bool}
        :param local: True if you upload local images
        :return:
        {
            'dataset_id': str,
            'new_items': int,
            'updated_items': int,
            'ignored_items': int,
        }
        """
        return self._client.populate_dataset(
            self.id, dataset_items, batch_size=batch_size
        )

    def iloc(self, i: int) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Number.
        :param i: absolute number of dataset item for the given dataset.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Box2DAnnotation],
        }
        """
        return self._client.dataitem_iloc(self.id, i)

    def refloc(self, reference_id: str) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Reference Id.
        :param reference_id: reference_id of dataset item.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Box2DAnnotation],
        }
        """
        return self._client.dataitem_ref_id(self.id, reference_id)

    def loc(self, dataset_item_id: str) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Id.
        :param dataset_item_id: internally controlled id for the dataset item.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Box2DAnnotation],
        }
        """
        return self._client.dataitem_loc(self.id, dataset_item_id)

    def create_slice(
        self,
        name: str,
        dataset_item_ids: List[str] = None,
        reference_ids: List[str] = None,
    ):
        """
        Creates a slice from items already present in a dataset.
        The caller must exclusively use either datasetItemIds or reference_ids
        as a means of identifying items in the dataset.

        "name" -- The human-readable name of the slice.

        "dataset_item_ids" -- An optional list of dataset item ids for the items in the slice

        "reference_ids" -- An optional list of user-specified identifier for the items in the slice

        :param
        payload:
        {
            "name": str,
            "dataset_item_ids": List[str],
            "reference_ids": List[str],
        }
        :return: new Slice object
        """
        if dataset_item_ids and reference_ids:
            raise Exception(
                "You cannot both dataset_item_ids and reference_ids"
            )
        payload: Dict[str, Any] = {NAME_KEY: name}
        if dataset_item_ids:
            payload[DATASET_ITEM_IDS_KEY] = dataset_item_ids
        if reference_ids:
            payload[REFERENCE_IDS_KEY] = reference_ids
        return self._client.create_slice(self.id, payload)
