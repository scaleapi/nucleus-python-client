from typing import Any, Dict, List, Optional, Sequence, Union

from dataclasses import dataclass
import requests

from nucleus.job import AsyncJob
from nucleus.prediction import (
    BoxPrediction,
    CuboidPrediction,
    PolygonPrediction,
    SegmentationPrediction,
    from_json,
)
from nucleus.url_utils import sanitize_string_args
from nucleus.utils import (
    convert_export_payload,
    format_dataset_item_response,
    format_prediction_response,
    serialize_and_write_to_presigned_url,
)

from .annotation import Annotation, check_all_mask_paths_remote
from .constants import (
    ANNOTATIONS_KEY,
    AUTOTAG_SCORE_THRESHOLD,
    DATASET_LENGTH_KEY,
    DATASET_MODEL_RUNS_KEY,
    DATASET_NAME_KEY,
    DATASET_SLICES_KEY,
    DEFAULT_ANNOTATION_UPDATE_MODE,
    EXPORTED_ROWS,
    NAME_KEY,
    REFERENCE_IDS_KEY,
    REQUEST_ID_KEY,
    UPDATE_KEY,
)
from .dataset_item import (
    DatasetItem,
    check_all_paths_remote,
    check_for_duplicate_reference_ids,
)
from .payload_constructor import (
    construct_append_scenes_payload,
    construct_model_run_creation_payload,
    construct_taxonomy_payload,
)
from .scene import LidarScene, Scene, check_all_scene_paths_remote

WARN_FOR_LARGE_UPLOAD = 50000
WARN_FOR_LARGE_SCENES_UPLOAD = 5


@dataclass
class Dataset:
    """Datasets are collections of your data that can be associated with models.

    You can append :class:`DatasetItems<DatasetItem>` or :class:`Scenes<LidarScene>`
    with metadata to your dataset, annotate it with ground truth, and upload
    model predictions to evaluate and compare model performance on you data.

    Datasets cannot be instantiated directly and instead must be created via API
    endpoint, using :meth:`NucleusClient.create_dataset` or similar.
    """

    id: str
    _client: "NucleusClient"

    def __repr__(self):
        return f"Dataset(dataset_id='{self.id}', client={self._client})"

    def __eq__(self, other):
        if self.id == other.id:
            if self._client == other._client:
                return True
        return False

    @property
    def name(self) -> str:
        """User-defined name of the Dataset."""
        return self.info().get(DATASET_NAME_KEY, "")

    @property
    def model_runs(self) -> List[str]:
        """List of all model runs associated with the Dataset."""
        # TODO: model_runs -> models
        return self.info().get(DATASET_MODEL_RUNS_KEY, [])

    @property
    def slices(self) -> List[str]:
        """List of all Slice IDs created from the Dataset."""
        return self.info().get(DATASET_SLICES_KEY, [])

    @property
    def size(self) -> int:
        """Number of items in the Dataset."""
        return self.info().get(DATASET_LENGTH_KEY, 0)

    @property
    def items(self) -> List[DatasetItem]:
        """List of all DatasetItem objects in the Dataset."""
        return self._client.get_dataset_items(self.id)

    @sanitize_string_args
    def autotag_items(self, autotag_name, for_scores_greater_than=0):
        """Fetches the autotag's items above the score threshold, sorted by descending score.

        Parameters:
            autotag_name: The user-defined name of the autotag.
            for_scores_greater_than (int, optional): Score threshold between -1
                and 1 above which to include autotag items.

        Returns:
            List of autotagged items above the given score threshold, sorted by
            descending score, and autotag info, packaged into a dict as follows::

                {
                    "autotagItems": List[{
                        ref_id: str,
                        score: float,
                        model_prediction_annotation_id: str | None
                        ground_truth_annotation_id: str | None,
                    }],
                    "autotag": {
                        id: str,
                        name: str,
                        status: "started" | "completed",
                        autotag_level: "Image" | "Object"
                    }
                }

            Note ``model_prediction_annotation_id`` and ``ground_truth_annotation_id``
            are only relevant for object autotags.
        """
        response = self._client.make_request(
            payload={AUTOTAG_SCORE_THRESHOLD: for_scores_greater_than},
            route=f"dataset/{self.id}/autotag/{autotag_name}/taggedItems",
            requests_command=requests.get,
        )
        return response

    def autotag_training_items(self, autotag_name):
        """Fetches items that were manually selected during refinement of the autotag.

        Parameters:
            autotag_name: The user-defined name of the autotag.

        Returns:
            List of user-selected positives and autotag info, packaged into a
            dict as follows::

                {
                    "autotagPositiveTrainingItems": {
                        ref_id: str,
                        model_prediction_annotation_id: str | None,
                        ground_truth_annotation_id: str | None,
                    }[],
                    "autotag": {
                        id: str,
                        name: str,
                        status: "started" | "completed",
                        autotag_level: "Image" | "Object"
                    }
                }

            Note ``model_prediction_annotation_id`` and ``ground_truth_annotation_id``
            are only relevant for object autotags.
        """
        response = self._client.make_request(
            payload={},
            route=f"dataset/{self.id}/autotag/{autotag_name}/trainingItems",
            requests_command=requests.get,
        )
        return response

    def info(self) -> dict:
        """Retrieve information about the dataset.

        Returns:
            Payload containing information and members of the dataset::

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
        # TODO: deprecate ModelRun
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
        """Uploads ground truth annotations to the dataset.

        Adding ground truth to your dataset in Nucleus allows you to visualize annotations,
        query dataset items based on the annotations they contain, and evaluate ModelRuns by
        comparing predictions to ground truth.

        Nucleus supports :class:`Box<BoxAnnotation>`, :class:`Polygon<PolygonAnnotation>`,
        :class:`Cuboid<CuboidAnnotation>`, :class:`Segmentation<SegmentationAnnotation>`,
        and :class:`Category<CategoryAnnotation>` annotations. Cuboid annotations
        can only be uploaded to a :class:`pointcloud DatasetItem<LidarScene>`.

        When uploading an annotation, you need to specify which item you are annotating via
        the reference_id you provided when uploading the image or pointcloud.

        Ground truth uploads can be made idempotent by specifying an optional annotation_id for
        each annotation. This id should be unique within the dataset_item so that
        (reference_id, annotation_id) is unique within the dataset.

        When uploading a mask annotation, Nucleus expects the mask file to be in PNG format
        with each pixel being a 0-255 uint8. Currently, Nucleus only supports uploading masks
        from URL.

        Nucleus automatically enforces the constraint that each DatasetItem can have at most one
        ground truth segmentation mask. As a consequence, if during upload a duplicate mask is
        detected for a given image, by default it will be ignored. You can change this behavior
        by specifying the optional 'update' flag. Setting update = true will replace the
        existing segmentation with the new mask specified in the request body.

        For ingesting large datasets, see the Guide for Large Ingestions.

        .. todo ::
            add link to Guide for Large Ingestions

        Parameters:
            annotations (Sequence[:class:`Annotation`]): List of annotation objects to upload.
            update: Whether to ignore or update metadata for conflicting annotations.
            batch_size: Number of annotations processed in each concurrent batch.
              Default is 5000.
            asynchronous: Whether or not to process the upload asynchronously (and
              return an :class:`AsyncJob` object). Default is False.

        Returns:
            If synchronous, payload describing the upload result::
                
                {
                    "dataset_id": str,
                    "annotations_processed": int
                }

            Otherwise, returns an :class:`AsyncJob` object.
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

    def ingest_tasks(self, task_ids: List[str]) -> dict:
        """Ingest specific tasks from an existing Scale or Rapid project into the dataset.

        Parameters:
            task_ids: List of task IDs to ingest.

        Returns:
            Payload describing the asynchronous upload result::
            
                {
                    "ingested_tasks": int,
                    "ignored_tasks": int,
                    "pending_tasks": int
                }
        """
        return self._client.ingest_tasks(self.id, {"tasks": task_ids})

    def append(
        self,
        items: Union[Sequence[DatasetItem], Sequence[LidarScene]],
        update: Optional[bool] = False,
        batch_size: Optional[int] = 20,
        asynchronous=False,
    ) -> Union[dict, AsyncJob]:
        """Appends items or scenes to a dataset.

        ::

            import nucleus

            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            dataset = client.get_dataset("ds_bw6de8s84pe0vbn6p5zg")

            local_item = nucleus.DatasetItem(
              image_location="./1.jpg",
              reference_id="image_1",
              metadata={"key": "value"}
            )
            remote_item = nucleus.DatasetItem(
              image_location="s3://your-bucket/2.jpg",
              reference_id="image_2",
              metadata={"key": "value"}
            )

            # default is synchronous upload
            sync_response = dataset.append(items=[local_item])

            # async jobs have higher throughput but can be more difficult to debug
            async_job = dataset.append(
              items=[remote_item], # all items must be remote for async
              asynchronous=True
            )
            print(async_job.status())

        A :class:`Dataset` can be populated with labeled and unlabeled
        data. Using Nucleus, you can filter down the data inside your dataset
        using custom metadata about your images.

        For instance, your local dataset may contain ``Sunny``, ``Foggy``, and
        ``Rainy`` folders of images. All of these images can be uploaded into a
        single Nucleus ``Dataset``, with (queryable) metadata like ``{"weather":
        "Sunny"}``.

        To update an item's metadata, you can re-ingest the same items with the
        ``update`` argument set to true. Existing metadata will be overwritten
        for ``DatasetItems`` in the payload that share a ``reference_id`` with a
        previously uploaded ``DatasetItem``. To retrieve your existing
        ``reference_ids``, use :meth:`Dataset.items`.

        ::

            # update metadata by reuploading the item
            remote_item.metadata["weather"] = "Sunny"

            async_job_2 = dataset.append(
              items=[remote_item],
              update=True,
              asynchronous=True
            )

        Parameters:
            dataset_items ( \
              Union[ \
                Sequence[:class:`DatasetItem`], \
                Sequence[:class:`LidarScene`] \
              ]): List of items or scenes to upload.
            batch_size: Size of the batch for larger uploads. Default is 20.
            update: Whether or not to update metadata on reference ID collision.
              Default is False.
            asynchronous: Whether or not to process the upload asynchronously (and
              return an :class:`AsyncJob` object). This is highly encouraged for
              3D data to drastically increase throughput. Default is False.

        Returns:
            If synchronous, returns a payload describing the upload result::

                {
                    "dataset_id: str,
                    "new_items": int,
                    "updated_items": int,
                    "ignored_items": int,
                    "upload_errors": int
                }

            Otherwise, returns an :class:`AsyncJob` object.
        """
        assert (
            batch_size is None or batch_size < 30
        ), "Please specify a batch size smaller than 30 to avoid timeouts."
        dataset_items = [
            item for item in items if isinstance(item, DatasetItem)
        ]
        scenes = [item for item in items if isinstance(item, LidarScene)]
        if dataset_items and scenes:
            raise Exception(
                "You must append either DatasetItems or Scenes to the dataset."
            )
        if scenes:
            assert (
                asynchronous
            ), "In order to avoid timeouts, you must set asynchronous=True when uploading scenes."
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
        # TODO: make private in favor of Dataset.append invocation
        for scene in scenes:
            scene.validate()

        if not asynchronous:
            print(
                "WARNING: Processing lidar pointclouds usually takes several seconds. As a result, sychronous scene upload"
                "requests are likely to timeout. For large uploads, we recommend using the flag asynchronous=True "
                "to avoid HTTP timeouts. Please see"
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
        """Retrieves dataset item by absolute numerical index.

        Parameters:
            i: Absolute numerical index of the dataset item within the dataset.

        Returns:
            Payload describing the dataset item and associated members::

                {
                    "item": {
                        "metadata": Optional[dict],
                        "original_image_url": str,
                        "reference_id": str,
                        "dataset_item_id": str
                    },
                    "annotations": List[{
                        "<annotation_type>": {
                            "id": str,
                            "dataset_item_id": str,
                            "label": str,
                            "geometry": Optional[dict],
                            "mask_url": Optional[str],
                            "annotations": Optional[{
                                "index": int,
                                "label": str
                            }],
                            "annotation_id": Optional[str],
                            "metadata": Optional[dict],
                            "type": "<annotation_type>",
                            "taxonomy_name": Optional[str],
                            "reference_id": str
                        }]
                    },
                    "task_ids": List[str]
                }
        """
        response = self._client.dataitem_iloc(self.id, i)
        return format_dataset_item_response(response)

    def refloc(self, reference_id: str) -> dict:
        """Retrieves a dataset item by reference ID.

        Parameters:
            reference_id: User-defined reference ID of the dataset item.

        Returns:
            Payload describing the dataset item and associated members::

                {
                    "item": {
                        "metadata": Optional[dict],
                        "original_image_url": str,
                        "reference_id": str,
                        "dataset_item_id": str
                    },
                    "annotations": List[{
                        "<annotation_type>": {
                            "id": str,
                            "dataset_item_id": str,
                            "label": str,
                            "geometry": Optional[dict],
                            "mask_url": Optional[str],
                            "annotations": Optional[{
                                "index": int,
                                "label": str
                            }],
                            "annotation_id": Optional[str],
                            "metadata": Optional[dict],
                            "type": "<annotation_type>",
                            "taxonomy_name": Optional[str],
                            "reference_id": str
                        }]
                    },
                    "task_ids": List[str]
                }
        """
        response = self._client.dataitem_ref_id(self.id, reference_id)
        return format_dataset_item_response(response)

    def loc(self, dataset_item_id: str) -> dict:
        """Retrieves a dataset item by Nucleus-generated ID.

        Parameters:
            dataset_item_id: Nucleus-generated dataset item ID (starts with ``di_``).
              This can be retrieved via :meth:`Dataset.items` or a Nucleus dashboard URL.

        Returns:
            Payload describing the dataset item and associated members::

                {
                    "item": {
                        "metadata": Optional[dict],
                        "original_image_url": str,
                        "reference_id": str,
                        "dataset_item_id": str
                    },
                    "annotations": List[{
                        "<annotation_type>": {
                            "id": str,
                            "dataset_item_id": str,
                            "label": str,
                            "geometry": Optional[dict],
                            "mask_url": Optional[str],
                            "annotations": Optional[{
                                "index": int,
                                "label": str
                            }],
                            "annotation_id": Optional[str],
                            "metadata": Optional[dict],
                            "type": "<annotation_type>",
                            "taxonomy_name": Optional[str],
                            "reference_id": str
                        }]
                    },
                    "task_ids": List[str]
                }
        """
        response = self._client.dataitem_loc(self.id, dataset_item_id)
        return format_dataset_item_response(response)

    def ground_truth_loc(self, reference_id: str, annotation_id: str):
        """Fetches a single ground truth annotation by id.

        Parameters:
            reference_id: User-defined reference ID of the dataset item associated
              with the ground truth annotation.
            annotation_id: User-defined ID of the ground truth annotation.

        Returns:
            Union[\
                :class:`BoxAnnotation`, \
                :class:`PolygonAnnotation`, \
                :class:`CuboidAnnotation`, \
                :class:`SegmentationAnnotation` \
                :class:`CategoryAnnotation` \
            ]: Ground truth annotation object with the specified annotation ID.
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
        """Creates a :class:`Slice` of dataset items within a dataset.
        
        Parameters:
            name: A human-readable name for the slice.
            reference_ids: List of reference IDs of dataset items to add to the slice::

        Returns:
            :class:`Slice`: The newly constructed slice item.
        """
        return self._client.create_slice(
            self.id, {NAME_KEY: name, REFERENCE_IDS_KEY: reference_ids}
        )

    def delete_item(self, reference_id: str) -> dict:
        """Deletes an item from the dataset by item reference ID.

        All annotations and predictions associated with the item will be deleted
        as well.

        Parameters:
            reference_id: The user-defined reference ID of the item to delete.

        Returns:
            Payload to indicate deletion invocation.
        """
        return self._client.delete_dataset_item(
            self.id, reference_id=reference_id
        )

    def list_autotags(self):
        """Fetches all autotags of the dataset.

        Returns:
            List of autotag payloads::
                
                List[{
                    "id": str,
                    "name": str,
                    "status": "completed" | "pending",
                    "autotag_level": "Image" | "Object"
                }]
        """
        return self._client.list_autotags(self.id)

    def create_custom_index(self, embeddings_urls: List[str], embedding_dim: int):
        """Processes user-provided embeddings for the dataset to use with autotag.

        ::

            import nucleus

            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            dataset = client.get_dataset("ds_bw6de8s84pe0vbn6p5zg")

            embeddings = {
                "reference_id_0": [0.1, 0.2, 0.3],
                "reference_id_1": [0.4, 0.5, 0.6],
            } # uploaded to s3 with the below URL

            embeddings_url = "s3://dataset/embeddings_map.json"

            response = dataset.create_custom_index(
                embeddings_url=[embeddings_url],
                embedding_dim=3
            )

        Parameters:
            embeddings_urls:  List of URLs, each of which pointing to
              a JSON mapping reference_id -> embedding vector.
            embedding_dim: The dimension of the embedding vectors. Must be consistent
              across all embedding vectors in the index.

        Returns:
            :class:`AsyncJob`: Asynchronous job object to track processing status.
        """
        return AsyncJob.from_json(
            self._client.create_custom_index(
                self.id,
                embeddings_urls,
                embedding_dim,
            ),
            self._client,
        )

    def delete_custom_index(self):
        """Deletes the custom index uploaded to the dataset.

        Returns:
            Payload containing information that can be used to track the job's status::

                {
                    "dataset_id": str,
                    "job_id": str,
                    "message": str
                }
        """
        return self._client.delete_custom_index(self.id)

    def set_continuous_indexing(self, enable: bool = True):
        """Toggle whether embeddings are automatically generated for new data.

        Sets continuous indexing for a given dataset, which will automatically
        generate embeddings for use with autotag whenever new images are uploaded.

        Parameters:
            enable: Whether to enable or disable continuous indexing. Default is
              True.

        Returns:
            Response payload::
                
                {
                    "dataset_id": str,
                    "message": str
                }
        """
        return self._client.set_continuous_indexing(self.id, enable)

    def create_image_index(self):
        """Creates or updates image index by generating embeddings for lacking images.

        The embeddings are used for autotag and similarity search.

        This endpoint is limited to index up to 2 million images at a time and the
        job will fail for payloads that exceed this limit.

        Response:
            :class:`AsyncJob`: Asynchronous job object to track processing status.
        """
        response = self._client.create_image_index(self.id)
        return AsyncJob.from_json(response, self._client)

    def create_object_index(
        self, model_run_id: str = None, gt_only: bool = None
    ):
        """Creates or updates object index by generating embeddings for lacking objects.

        These embeddings are used for autotag and similarity search. This endpoint
        only supports indexing objects sourced from the predictions of a specific
        model or the ground truth annotations of the dataset.

        This endpoint is idempotent. If this endpoint is called again for a model
        whose predictions were indexed in the past, the previously indexed predictions
        will not have new embeddings recomputed. The same is true for ground truth
        annotations.

        Note that this means if you change update a prediction or ground truth
        bounding box that already has an associated embedding, the embedding will
        not be updated, even with another call to this endpoint. For now, we
        recommend deleting the prediction or ground truth annotation and
        re-inserting it to force generate a new embedding.

        This endpoint is limited to generating embeddings for 3 million objects
        at a time and the job will fail for payloads that exceed this limit.

        Parameters:
            model_run_id: The ID of the model whose predictions should be indexed.
              Default is None, but must be supplied in the absence of ``gt_only``.
                
              .. todo ::
                  Deprecate model run

            gt_only: Whether to only generate embeddings for the ground truth
              annotations of the dataset. Default is None, but must be supplied
              in the absence of ``model_run_id``.

        Returns:
            Payload containing an :class:`AsyncJob` object to monitor progress.
        """
        response = self._client.create_object_index(
            self.id, model_run_id, gt_only
        )
        return AsyncJob.from_json(response, self._client)

    def add_taxonomy(
        self,
        taxonomy_name: str,
        taxonomy_type: str,
        labels: List[str],
    ):
        """Creates a new taxonomy.
        ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            dataset = client.get_dataset("ds_bw6de8s84pe0vbn6p5zg")

            response = dataset.add_taxonomy(
                taxonomy_name="clothing_type",
                taxonomy_type="category",
                labels=["shirt", "trousers", "dress"]
            )

        Parameters:
            taxonomy_name: The name of the taxonomy. Taxonomy names must be
              unique within a dataset.
            taxonomy_type: The type of this taxonomy as a string literal.
              Currently, the only supported taxonomy type is "category".
            labels: The list of possible labels for the taxonomy.

        Returns:
            Returns a response with dataset_id, taxonomy_name and type for the
            new taxonomy.
        """
        return self._client.make_request(
            construct_taxonomy_payload(taxonomy_name, taxonomy_type, labels),
            f"dataset/{self.id}/add_taxonomy",
            requests_command=requests.post,
        )

    def items_and_annotations(
        self,
    ) -> List[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Returns a list of all DatasetItems and Annotations in this slice.

        Returns:
            A list of dicts, each with two keys representing a row in the dataset::
                
                List[{
                    "item": DatasetItem,
                    "annotations": {
                        "box": Optional[List[BoxAnnotation]],
                        "cuboid": Optional[List[CuboidAnnotation]],
                        "polygon": Optional[List[PolygonAnnotation]],
                        "segmentation": Optional[List[SegmentationAnnotation]],
                        "categorization": Optional[List[CategoryAnnotation]],
                    }
                }]
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
        """Fetches a pd.DataFrame-ready list of dataset embeddings.

        Returns:
            A list, where each item is a dict with two keys representing a row
            in the dataset::

                List[{
                    "reference_id": str,
                    "embedding_vector": List[float]
                }]
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
        """Deletes all annotations associated with the specified item reference IDs.

        Parameters:
            reference_ids: List of user-defined reference IDs of the dataset items
              from which to delete annotations.
            keep_history: Whether to preserve version history. If False, all
                previous versions will be deleted along with the annotations. If
                True, the version history (including deletion) wil persist.
                Default is False.

        Returns:
            Empty payload response.
        """
        response = self._client.delete_annotations(
            self.id, reference_ids, keep_history
        )
        return AsyncJob.from_json(response, self._client)

    def get_scene(self, reference_id: str) -> Scene:
        """Fetches a single scene in the dataset by its reference ID.

        Parameters:
            reference_id: User-defined reference ID of the scene.

        Returns:
            :class:`Scene<LidarScene>`: A scene object containing frames, which
            in turn contain pointcloud or image items.
        """
        return Scene.from_json(
            self._client.make_request(
                payload=None,
                route=f"dataset/{self.id}/scene/{reference_id}",
                requests_command=requests.get,
            )
        )

    def export_predictions(self, model):
        """Fetches all predictions of a model that were uploaded to the dataset.

        Parameters:
            model (:class:`Model`): The model whose predictions to retrieve.

        Returns:
            List[Union[\
                :class:`BoxPrediction`, \
                :class:`PolygonPrediction`, \
                :class:`CuboidPrediction`, \
                :class:`SegmentationPrediction` \
            ]]: List of prediction objects from the model.

        """
        json_response = self._client.make_request(
            payload=None,
            route=f"dataset/{self.id}/model/{model.id}/export",
            requests_command=requests.get,
        )
        return format_prediction_response({ANNOTATIONS_KEY: json_response})

    def calculate_evaluation_metrics(self, model, options: dict = None):
        """Starts computation of evaluation metrics for a model on the dataset.

        To update matches and metrics calculated for a model on a given dataset you
        can call this endpoint. This is required in order to sort by IOU, view false
        positives/false negatives, and view model insights.

        You can add predictions from a model to a dataset after running the
        calculation of the metrics. However, the calculation of metrics will have
        to be retriggered for the new predictions to be matched with ground truth
        and appear as false positives/negatives, or for the new predictions effect
        on metrics to be reflected in model run insights.

        During IoU calculation, bounding box Predictions are compared to
        GroundTruth using a greedy matching algorithm that matches prediction and
        ground truth boxes that have the highest ious first. By default the
        matching algorithm is class-sensitive: it will treat a match as a true
        postive only if the labels are the same.

        The algorithm can be tuned to classify true positives between certain
        classes, but not others. This is useful if the labels in your ground truth
        do not match the exact strings of your model predictions, or if you want
        to associate multiple predictions with one ground truth label, or multiple
        ground truth labels with one prediction. To recompute metrics based on
        different matching, you can re-commit the run with new request parameters.

        Parameters:
            model (:class:`Model`): The model object for which to calculate metrics.
            options: Dictionary of specific options to configure metrics calculation.

                class_agnostic
                  Whether ground truth and prediction classes can differ when
                  being matched for evaluation metrics. Default is True.

                allowed_label_matches
                  Pairs of ground truth and prediction classes that should
                  be considered matchable when computing metrics. If supplied,
                  ``class_agnostic`` must be False.
              
                ::

                    {
                        "class_agnostic": bool,
                        "allowed_label_matches": List[{
                            "ground_truth_label": str,
                            "model_prediction_label": str
                        }]
                    }
        """
        if options is None:
            options = {}
        return self._client.make_request(
            payload=options,
            route=f"dataset/{self.id}/model/{model.id}/calculateEvaluationMetrics",
        )

    def upload_predictions(
        self,
        model,
        predictions: List[
            Union[
                BoxPrediction,
                PolygonPrediction,
                CuboidPrediction,
                SegmentationPrediction,
            ]
        ], update: bool = False,
        asynchronous: bool = False,
    ):
        """Uploads predictions and associates them with an existing :class:`Model`.

        Parameters:
            model (:class:`Model`): Nucleus-generated model ID (starts with ``prj_``). This can
              be retrieved via :meth:`list_models` or a Nucleus dashboard URL.
            predictions (List[Union[\
                :class:`BoxPrediction`, \
                :class:`PolygonPrediction`, \
                :class:`CuboidPrediction`, \
                :class:`SegmentationPrediction` \
            ]]): List of prediction objects to upload.

                .. todo ::
                    Add CategoryPrediction to above typehints once supported

            update: Whether or not to update metadata or ignore on reference ID
              collision. Default is False.
            asynchronous: Whether or not to process the upload asynchronously (and
              return an :class:`AsyncJob` object). Default is False.

        Returns:
            Payload describing the synchronous upload::

                {
                    "dataset_id": str,
                    "model_run_id": str,
                    "predictions_processed": int,
                    "predictions_ignored": int,
                }
        """
        if asynchronous:
            check_all_mask_paths_remote(predictions)

            request_id = serialize_and_write_to_presigned_url(
                predictions, self.id, self._client
            )
            response = self._client.make_request(
                payload={REQUEST_ID_KEY: request_id, UPDATE_KEY: update},
                route=f"dataset/{self.id}/model/{model.id}/uploadPredictions?async=1",
            )
            return AsyncJob.from_json(response, self._client)
        else:
            return self._client.predict(
                model_run_id=None,
                dataset_id=self.id,
                model_id=model.id,
                annotations=predictions,
                update=update,
            )

    def predictions_iloc(self, model, index):
        """Fetches all predictions of a dataset item by its absolute index.

        Parameters:
            model (:class:`Model`): Model object from which to fetch the prediction.
            index (int): Absolute index of the dataset item within the dataset.

        Returns:
            Dict[str, List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction,
            SegmentationPrediction, CategoryPrediction]]]: Dictionary mapping prediction
            type to a list of such prediction objects from the given model::

                {
                    "box": List[BoxPrediction],
                    "polygon": List[PolygonPrediction],
                    "cuboid": List[CuboidPrediction],
                    "segmentation": List[SegmentationPrediction],
                    "category": List[CategoryPrediction],
                }
        """
        return format_prediction_response(
            self._client.make_request(
                payload=None,
                route=f"dataset/{self.id}/model/{model.id}/iloc/{index}",
                requests_command=requests.get,
            )
        )

    def predictions_refloc(self, model, reference_id):
        """Fetches all predictions of a dataset item by its reference ID.

        Parameters:
            model (:class:`Model`): Model object from which to fetch the prediction.
            reference_id (str): User-defined ID of the dataset item from which to fetch
              all predictions.

        Returns:
            Dict[str, List[Union[BoxPrediction, PolygonPrediction, CuboidPrediction,
            SegmentationPrediction, CategoryPrediction]]]: Dictionary mapping prediction
            type to a list of such prediction objects from the given model::

                {
                    "box": List[BoxPrediction],
                    "polygon": List[PolygonPrediction],
                    "cuboid": List[CuboidPrediction],
                    "segmentation": List[SegmentationPrediction],
                    "category": List[CategoryPrediction],
                }
        """
        return format_prediction_response(
            self._client.make_request(
                payload=None,
                route=f"dataset/{self.id}/model/{model.id}/referenceId/{reference_id}",
                requests_command=requests.get,
            )
        )

    def prediction_loc(self, model, reference_id, annotation_id):
        """Fetches a single ground truth annotation by id.

        Parameters:
            model (:class:`Model`): Model object from which to fetch the prediction.
            reference_id (str): User-defined reference ID of the dataset item
              associated with the model prediction.
            annotation_id (str): User-defined ID of the ground truth annotation.

        Returns:
            Union[\
                :class:`BoxPrediction`, \
                :class:`PolygonPrediction`, \
                :class:`CuboidPrediction`, \
                :class:`SegmentationPrediction` \
                :class:`CategoryPrediction` \
            ]: Model prediction object with the specified annotation ID.
        """
        return from_json(
            self._client.make_request(
                payload=None,
                route=f"dataset/{self.id}/model/{model.id}/loc/{reference_id}/{annotation_id}",
                requests_command=requests.get,
            )
        )
