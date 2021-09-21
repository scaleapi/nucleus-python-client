from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from nucleus.job import AsyncJob
from nucleus.url_utils import sanitize_string_args
from nucleus.utils import (
    convert_export_payload,
    format_dataset_item_response,
    serialize_and_write_to_presigned_url,
)

from .annotation import (
    Annotation,
    check_all_mask_paths_remote,
)
from .constants import (
    DATASET_LENGTH_KEY,
    DATASET_MODEL_RUNS_KEY,
    DATASET_NAME_KEY,
    DATASET_SLICES_KEY,
    DEFAULT_ANNOTATION_UPDATE_MODE,
    EXPORTED_ROWS,
    NAME_KEY,
    REFERENCE_IDS_KEY,
    REQUEST_ID_KEY,
    AUTOTAG_SCORE_THRESHOLD,
    UPDATE_KEY,
)
from .dataset_item import (
    DatasetItem,
    check_all_paths_remote,
    check_for_duplicate_reference_ids,
)
from .scene import LidarScene, check_all_scene_paths_remote
from .payload_constructor import (
    construct_append_scenes_payload,
    construct_model_run_creation_payload,
    construct_taxonomy_payload,
)

WARN_FOR_LARGE_UPLOAD = 50000
WARN_FOR_LARGE_SCENES_UPLOAD = 5


class Dataset:
    """
    Nucleus Dataset. You can append images with metadata to your dataset,
    annotate it with ground truth and upload model predictions to evaluate and
    compare model performance on you data.
    """

    def __init__(
        self,
        dataset_id: str,
        client: "NucleusClient",  # type:ignore # noqa: F821
    ):
        self.id = dataset_id
        self._client = client

    def __repr__(self):
        return f"Dataset(dataset_id='{self.id}', client={self._client})"

    def __eq__(self, other):
        if self.id == other.id:
            if self._client == other._client:
                return True
        return False

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

    @sanitize_string_args
    def autotag_items(self, autotag_name, for_scores_greater_than=0):
        """For a given Autotag of this dataset, export its tagged items with scores above a threshold, largest scores first.

        :return: dictionary of the form
            {
                'autotagItems': {
                    ref_id: str,
                    score: float,
                    model_prediction_annotation_id: str | None
                    ground_truth_annotation_id: str | None,
                }[],
                'autotag': {
                    id: str,
                    name: str,
                    status: 'started' | 'completed',
                    autotag_level: 'Image' | 'Object'
                }
            }
        See https://dashboard.nucleus.scale.com/nucleus/docs/api#export-autotag-items for more details on the return types.
        """
        response = self._client.make_request(
            payload={AUTOTAG_SCORE_THRESHOLD: for_scores_greater_than},
            route=f"dataset/{self.id}/autotag/{autotag_name}/taggedItems",
            requests_command=requests.get,
        )
        return response

    def autotag_training_items(self, autotag_name):
        """For a given Autotag of this dataset, export its training items. These are user selected positives during refinement.

        :return: dictionary of the form
            {
                'autotagPositiveTrainingItems': {
                    ref_id: str,
                    model_prediction_annotation_id: str | None,
                    ground_truth_annotation_id: str | None,
                }[],
                'autotag': {
                    id: str,
                    name: str,
                    status: 'started' | 'completed',
                    autotag_level: 'Image' | 'Object'
                }
            }
        See https://dashboard.nucleus.scale.com/nucleus/docs/api#export-autotag-training-items for more details on the return types.
        """
        response = self._client.make_request(
            payload={},
            route=f"dataset/{self.id}/autotag/{autotag_name}/trainingItems",
            requests_command=requests.get,
        )
        return response

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

    def create_model_run(
        self,
        name: str,
        reference_id: Optional[str] = None,
        model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        annotation_metadata_schema: Optional[Dict] = None,
    ):
        """
        :param name: A name for the model run.
        :param reference_id: The user-specified reference identifier to associate with the model.
                        The 'model_id' field should be empty if this field is populated,
        :param model_id: The internally-controlled identifier of the model.
                    The 'reference_id' field should be empty if this field is populated,
        :param metadata: An arbitrary metadata blob for the current run.
        :param annotation_metadata_schema: A dictionary that defines schema for annotations.
        :param segmentation_metadata_schema: A dictionary that defines schema for segmentation.

        :return:
        {
          "model_id": str,
          "model_run_id": str,
        }
        """
        payload = construct_model_run_creation_payload(
            name,
            reference_id,
            model_id,
            metadata,
            annotation_metadata_schema,
        )
        return self._client.create_model_run(self.id, payload)

    def annotate(
        self,
        annotations: Sequence[Annotation],
        update: Optional[bool] = DEFAULT_ANNOTATION_UPDATE_MODE,
        batch_size: int = 5000,
        asynchronous: bool = False,
    ) -> Union[Dict[str, Any], AsyncJob]:
        """
        Uploads ground truth annotations for a given dataset.
        :param annotations: ground truth annotations for a given dataset to upload
        :param batch_size: batch parameter for long uploads
        :return:
        {
            "dataset_id: str,
            "new_items": int,
            "updated_items": int,
            "ignored_items": int,
        }
        """
        check_all_mask_paths_remote(annotations)

        if asynchronous:
            request_id = serialize_and_write_to_presigned_url(
                annotations, self.id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"dataset/{self.id}/annotate?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        return self._client.annotate_dataset(
            self.id, annotations, update=update, batch_size=batch_size
        )

    def ingest_tasks(self, task_ids: dict):
        """
        If you already submitted tasks to Scale for annotation this endpoint ingests your completed tasks
        annotated by Scale into your Nucleus Dataset.
        Right now we support ingestion from Videobox Annotation and 2D Box Annotation projects.
        Lated we'll support more annotation types.
        :param task_ids: list of task ids
        :return: {"ingested_tasks": int, "ignored_tasks": int, "pending_tasks": int}
        """
        return self._client.ingest_tasks(self.id, {"tasks": task_ids})

    def append(
        self,
        items: Union[Sequence[DatasetItem], Sequence[LidarScene]],
        update: Optional[bool] = False,
        batch_size: Optional[int] = 20,
        asynchronous=False,
    ) -> Union[dict, AsyncJob]:
        """
        Appends images with metadata (dataset items) or scenes to the dataset. Overwrites images on collision if forced.

        Parameters:
        :param items: items to upload
        :param update: if True overwrites images and metadata on collision
        :param batch_size: batch parameter for long uploads
        :param aynchronous: if True, return a job object representing asynchronous ingestion job.
        :return:
        {
            'dataset_id': str,
            'new_items': int,
            'updated_items': int,
            'ignored_items': int,
        }
        """
        dataset_items = [
            item for item in items if isinstance(item, DatasetItem)
        ]
        scenes = [item for item in items if isinstance(item, LidarScene)]
        if dataset_items and scenes:
            raise Exception(
                "You must append either DatasetItems or Scenes to the dataset."
            )
        if scenes:
            return self.append_scenes(scenes, update, asynchronous)

        check_for_duplicate_reference_ids(dataset_items)

        if len(dataset_items) > WARN_FOR_LARGE_UPLOAD and not asynchronous:
            print(
                "Tip: for large uploads, get faster performance by importing your data "
                "into Nucleus directly from a cloud storage provider. See "
                "https://dashboard.scale.com/nucleus/docs/api?language=python#guide-for-large-ingestions"
                " for details."
            )

        if asynchronous:
            check_all_paths_remote(dataset_items)
            request_id = serialize_and_write_to_presigned_url(
                dataset_items, self.id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"dataset/{self.id}/append?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        return self._client.populate_dataset(
            self.id,
            dataset_items,
            update=update,
            batch_size=batch_size,
        )

    def append_scenes(
        self,
        scenes: List[LidarScene],
        update: Optional[bool] = False,
        asynchronous: Optional[bool] = False,
    ) -> Union[dict, AsyncJob]:
        """
        Appends scenes with given frames (containing pointclouds and optional images) to the dataset

        Parameters:
        :param scenes: scenes to upload
        :param update: if True, overwrite scene on collision
        :param asynchronous: if True, return a job object representing asynchronous ingestion job
        :return:
        {
            'dataset_id': str,
            'new_scenes': int,
            'ignored_scenes': int,
            'scenes_errored': int,
            'errors': List[str],
        }
        """
        for scene in scenes:
            scene.validate()

        if len(scenes) > WARN_FOR_LARGE_SCENES_UPLOAD and not asynchronous:
            print(
                "Tip: for large uploads, get faster performance by importing your data "
                "into Nucleus directly from a cloud storage provider. See "
                "https://dashboard.scale.com/nucleus/docs/api?language=python#guide-for-large-ingestions"
                " for details."
            )

        if asynchronous:
            check_all_scene_paths_remote(scenes)
            request_id = serialize_and_write_to_presigned_url(
                scenes, self.id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"{self.id}/upload_scenes?async=1",
            )
            return AsyncJob.from_json(response, self._client)

        payload = construct_append_scenes_payload(scenes, update)
        response = self._client.make_request(
            payload=payload,
            route=f"{self.id}/upload_scenes",
        )
        return response

    def iloc(self, i: int) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Number.
        :param i: absolute number of dataset item for the given dataset.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Union[BoxAnnotation, PolygonAnnotation, CuboidAnnotation, SegmentationAnnotation]],
        }
        """
        response = self._client.dataitem_iloc(self.id, i)
        return format_dataset_item_response(response)

    def refloc(self, reference_id: str) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Reference Id.
        :param reference_id: reference_id of dataset item.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Union[BoxAnnotation, PolygonAnnotation, CuboidAnnotation, SegmentationAnnotation]],
        }
        """
        response = self._client.dataitem_ref_id(self.id, reference_id)
        return format_dataset_item_response(response)

    def loc(self, dataset_item_id: str) -> dict:
        """
        Returns Dataset Item Info By Dataset Item Id.
        :param dataset_item_id: internally controlled id for the dataset item.
        :return:
        {
            "item": DatasetItem,
            "annotations": List[Union[BoxAnnotation, PolygonAnnotation, CuboidAnnotation, SegmentationAnnotation]],
        }
        """
        response = self._client.dataitem_loc(self.id, dataset_item_id)
        return format_dataset_item_response(response)

    def ground_truth_loc(self, reference_id: str, annotation_id: str):
        """
        Returns info for single ground truth Annotation by its id.
        :param reference_id: User specified id for the dataset item the ground truth is attached to
        :param annotation_id: User specified, or auto-generated id for the annotation
        :return:
        BoxAnnotation | PolygonAnnotation | CuboidAnnotation
        """
        response = self._client.make_request(
            {},
            f"dataset/{self.id}/groundTruth/loc/{reference_id}/{annotation_id}",
            requests.get,
        )
        return Annotation.from_json(response)

    def create_slice(
        self,
        name: str,
        reference_ids: List[str],
    ):
        """
        Creates a slice from items already present in a dataset.
        The caller must exclusively use either datasetItemIds or reference_ids
        as a means of identifying items in the dataset.

        :param name: The human-readable name of the slice.
        :param reference_ids: A list of user-specified identifier for the items in the slice

        :return: new Slice object
        """
        return self._client.create_slice(
            self.id, {NAME_KEY: name, REFERENCE_IDS_KEY: reference_ids}
        )

    def delete_item(self, reference_id: str):
        return self._client.delete_dataset_item(
            self.id, reference_id=reference_id
        )

    def list_autotags(self):
        return self._client.list_autotags(self.id)

    def create_custom_index(self, embeddings_urls: list, embedding_dim: int):
        return self._client.create_custom_index(
            self.id,
            embeddings_urls,
            embedding_dim,
        )

    def delete_custom_index(self):
        return self._client.delete_custom_index(self.id)

    def set_continuous_indexing(self, enable: bool = True):
        return self._client.set_continuous_indexing(self.id, enable)

    def create_image_index(self):
        response = self._client.create_image_index(self.id)
        return AsyncJob.from_json(response, self._client)

    def add_taxonomy(
        self,
        taxonomy_name: str,
        taxonomy_type: str,
        labels: List[str],
    ):
        """
        Creates a new taxonomy.
        Returns a response with dataset_id, taxonomy_name and type for the new taxonomy.
        :param taxonomy_name: name of the taxonomy
        :param type: type of the taxonomy
        :param labels: list of possible labels for the taxonomy
        """
        return self._client.make_request(
            construct_taxonomy_payload(taxonomy_name, taxonomy_type, labels),
            f"dataset/{self.id}/add_taxonomy",
            requests_command=requests.post,
        )

    def check_index_status(self, job_id: str):
        return self._client.check_index_status(job_id)

    def items_and_annotations(
        self,
    ) -> List[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Returns a list of all DatasetItems and Annotations in this slice.

        Returns:
            A list, where each item is a dict with two keys representing a row
            in the dataset.
            * One value in the dict is the DatasetItem, containing a reference to the
                item that was annotated.
            * The other value is a dictionary containing all the annotations for this
                dataset item, sorted by annotation type.
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"dataset/{self.id}/exportForTraining",
            requests_command=requests.get,
        )
        return convert_export_payload(api_payload[EXPORTED_ROWS])

    def export_embeddings(
        self,
    ) -> List[Dict[str, Union[str, List[float]]]]:
        """Returns a pd.Dataframe-ready format of dataset embeddings.

        Returns:
            A list, where each item is a dict with two keys representing a row
            in the dataset.
            * One value in the dict is the reference id
            * The other value is a list of the embedding values
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"dataset/{self.id}/embeddings",
            requests_command=requests.get,
        )
        return api_payload

    def delete_annotations(
        self, reference_ids: list = None, keep_history=False
    ):
        response = self._client.delete_annotations(
            self.id, reference_ids, keep_history
        )
        return AsyncJob.from_json(response, self._client)
