import warnings
from typing import Dict, Iterable, List, Set, Tuple, Union

import requests

from nucleus.annotation import Annotation
from nucleus.constants import EXPORTED_ROWS
from nucleus.dataset_item import DatasetItem
from nucleus.job import AsyncJob
from nucleus.utils import convert_export_payload, format_dataset_item_response


class Slice:
    """A Slice represents a subset of DatasetItems in your Dataset.

    Slices are subsets of your Dataset that unlock curation and exploration
    workflows.  Instead of thinking of your Datasets as collections of data, it
    is useful to think about them as a collection of Slices. For instance, your
    dataset may contain different weather scenarios, traffic conditions, or
    highway types.

    Perhaps your Models perform poorly on foggy weather scenarios; it is then
    useful to slice your dataset into a "foggy" slice, and fine-tune model
    performance on this slice until it reaches the performance you desire.

    Slices cannot be instantiated directly and instead must be created in the
    dashboard, or via API endpoint using :meth:`Dataset.create_slice`.

    ::

        import nucleus

        client = nucleus.NucleusClient(YOUR_SCALE_API_KEY)
        dataset = client.get_dataset("ds_bwkezj6g5c4g05gqp1eg")

        ref_ids = ["interesting_item_1", "interesting_item_2"]
        slice = dataset.create_slice(name="interesting", reference_ids=ref_ids)
    """

    def __init__(self, slice_id: str, client):
        self.id = slice_id
        self._slice_id = slice_id
        self._client = client
        self._dataset_id = None

    def __repr__(self):
        return f"Slice(slice_id='{self.id}', client={self._client})"

    def __eq__(self, other):
        if self.id == other.id:
            if self._client == other._client:
                return True
        return False

    @property
    def slice_id(self):
        warnings.warn(
            "Using Slice.slice_id is deprecated. Prefer using Slice.id",
            DeprecationWarning,
        )
        return self._slice_id

    @property
    def dataset_id(self):
        """The ID of the Dataset to which the Slice belongs."""
        if self._dataset_id is None:
            self.info()
        return self._dataset_id

    def info(self) -> dict:
        """Retrieves info and items of the Slice.

        Returns:
            A dict mapping keys to the corresponding info retrieved.
            ::

                {
                    "name": Union[str, int],
                    "dataset_id": str,
                    "dataset_items": List[{
                        "id": str,
                        "metadata": Dict[str, Union[str, int, float]],
                        "ref_id": str,
                        "original_image_url": str
                    }]
                }
        """
        info = self._client.slice_info(self.id)
        self._dataset_id = info["dataset_id"]
        return info

    def append(
        self,
        reference_ids: List[str] = None,
    ) -> dict:
        """Appends existing DatasetItems from a Dataset to a Slice.

        The endpoint expects a list of DatasetItem reference IDs which are set
        at upload time.

        Parameters:
            reference_ids: A list of user-specified IDs for DatasetItems you wish
              to append.

        Returns:
            Dict of the slice_id and the newly appended DatasetItem IDs. ::

                {
                    "slice_id": str,
                    "new_items": List[str]
                }
        """
        response = self._client.append_to_slice(
            slice_id=self.id,
            reference_ids=reference_ids,
        )
        return response

    def items_and_annotation_generator(
        self,
    ) -> Iterable[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Provides a generator of all DatasetItems and Annotations in the slice.

        Returns:
            Generator where each element is a dict containing the DatasetItem
            and all of its associated Annotations, grouped by type.
            ::

                Iterable[{
                    "item": DatasetItem,
                    "annotations": {
                        "box": List[BoxAnnotation],
                        "polygon": List[PolygonAnnotation],
                        "cuboid": List[CuboidAnnotation],
                        "segmentation": List[SegmentationAnnotation],
                        "category": List[CategoryAnnotation],
                    }
                }]
        """
        info = self.info()
        for item_metadata in info["dataset_items"]:
            yield format_dataset_item_response(
                self._client.dataitem_loc(
                    dataset_id=info["dataset_id"],
                    dataset_item_id=item_metadata["id"],
                )
            )

    def items_and_annotations(
        self,
    ) -> List[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Provides a list of all DatasetItems and Annotations in the Slice.

        Returns:
            List where each element is a dict containing the DatasetItem
            and all of its associated Annotations, grouped by type (e.g. box).
            ::

                List[{
                    "item": DatasetItem,
                    "annotations": {
                        "box": List[BoxAnnotation],
                        "polygon": List[PolygonAnnotation],
                        "cuboid": List[CuboidAnnotation],
                        "segmentation": List[SegmentationAnnotation],
                        "category": List[CategoryAnnotation],
                    }
                }]
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"slice/{self.id}/exportForTraining",
            requests_command=requests.get,
        )
        return convert_export_payload(api_payload[EXPORTED_ROWS])

    def send_to_labeling(self, project_id: str):
        """Send items in the Slice as tasks to a Scale labeling project.

        This endpoint submits the items of the Slice as tasks to a pre-existing
        Scale Annotation project uniquely identified by projectId. Only projects
        of type General Image Annotation are currently supported. Additionally,
        in order for task submission to succeed, the project must have task
        instructions and geometries configured as project-level parameters.  In
        order to create a project or set project parameters, you must use the
        Scale Annotation API, which is documented here: `Scale Annotation API
        Documentation <https://docs.scale.com/reference/project-overview>`_.
        When the newly created annotation tasks are annotated, the annotations
        will be automatically reflected in the Nucleus platform.

        For self-serve projects, user can choose to submit the slice as a
        calibration batch, which is recommended for brand new labeling projects.
        For more information about calibration batches, please reference
        `Overview of Self Serve Workflow
        <https://docs.scale.com/reference/batch-overview>`_. Note: A batch can
        be either a calibration batch or a self label batch, but not both.

        Note: Nucleus only supports bounding box, polygon, and line annotations.
        If the project parameters specify any other geometries (ellipses or
        points), those objects will be annotated, but they will not be reflected
        in Nucleus.

        Parameters:
            project_id: Scale-defined ID of the target annotation project.

        .. todo ::
            Add the below parameters, if needed.

            calibration_batch (Optional[bool]): Relevant to Scale Rapid projects
                only.  An optional boolean signaling whether to send as a
                "calibration batch" for taskers to preliminarily evaluate your
                project instructions and parameters.
            self_label_batch (Optional[bool]): Relevant to Scale Rapid projects
                only.  An optional boolean signaling whether to send as a
                "self-label batch," in which your team can label internally
                through Scale Rapid.
        """
        response = self._client.make_request(
            {}, f"slice/{self.id}/{project_id}/send_to_labeling"
        )
        return AsyncJob.from_json(response, self._client)

    def export_embeddings(
        self,
    ) -> List[Dict[str, Union[str, List[float]]]]:
        """Fetches a pd.DataFrame-ready list of slice embeddings.

        Returns:
            A list where each element is a columnar mapping::

                List[{
                    "reference_id": str,
                    "embedding_vector": List[float]
                }]
        """
        api_payload = self._client.make_request(
            payload=None,
            route=f"slice/{self.id}/embeddings",
            requests_command=requests.get,
        )
        return api_payload


def check_annotations_are_in_slice(
    annotations: List[Annotation], slice_to_check: Slice
) -> Tuple[bool, Set[str]]:
    """Checks whether the supplied Annotation objects exist in the supplied Slice.

    This endpoint checks whether each Annotation object's reference ID (of the
    parent DatasetItem) exists in the Slice.

    Args:
        annotations: Annnotations with ids referring to targets.
        slice: The slice to check against.

    Returns:
        A tuple of two elements.

        1. True if all Annotations are in the Slice, False otherwise;
        2. List of reference IDs not in the Slice.
    """
    info = slice_to_check.info()

    reference_ids_not_found_in_slice = {
        annotation.reference_id
        for annotation in annotations
        if annotation.reference_id is not None
    }.difference(
        {item_metadata["ref_id"] for item_metadata in info["dataset_items"]}
    )
    if reference_ids_not_found_in_slice:
        annotations_are_in_slice = False
    else:
        annotations_are_in_slice = True

    return (
        annotations_are_in_slice,
        reference_ids_not_found_in_slice,
    )
