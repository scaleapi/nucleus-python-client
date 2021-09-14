from typing import Dict, Iterable, List, Set, Tuple, Union

import requests

from nucleus.annotation import Annotation
from nucleus.dataset_item import DatasetItem
from nucleus.job import AsyncJob
from nucleus.utils import convert_export_payload, format_dataset_item_response
from nucleus.constants import (
    EXPORTED_ROWS,
)


class Slice:
    """
    Slice respesents a subset of your Dataset.
    """

    def __init__(self, slice_id: str, client):
        self.slice_id = slice_id
        self._client = client
        self._dataset_id = None

    def __repr__(self):
        return f"Slice(slice_id='{self.slice_id}', client={self._client})"

    def __eq__(self, other):
        if self.slice_id == other.slice_id:
            if self._client == other._client:
                return True
        return False

    @property
    def dataset_id(self):
        """The id of the dataset this slice belongs to."""
        if self._dataset_id is None:
            self.info()
        return self._dataset_id

    def info(self) -> dict:
        """
        This endpoint provides information about specified slice.

        :return:
        {
            "name": str,
            "dataset_id": str,
            "dataset_items",
        }
        """
        info = self._client.slice_info(self.slice_id)
        self._dataset_id = info["dataset_id"]
        return info

    def append(
        self,
        reference_ids: List[str] = None,
    ) -> dict:
        """
        Appends to a slice from items already present in a dataset.
        The caller must exclusively use either datasetItemIds or reference_ids
        as a means of identifying items in the dataset.

        :param
        reference_ids: List[str],

        :return:
        {
            "slice_id": str,
        }
        """
        response = self._client.append_to_slice(
            slice_id=self.slice_id,
            reference_ids=reference_ids,
        )
        return response

    def items_and_annotation_generator(
        self,
    ) -> Iterable[Dict[str, Union[DatasetItem, Dict[str, List[Annotation]]]]]:
        """Returns an iterable of all DatasetItems and Annotations in this slice.

        Returns:
            An iterable, where each item is a dict with two keys representing a row
            in the dataset.
            * One value in the dict is the DatasetItem, containing a reference to the
                item that was annotated, for example an image_url.
            * The other value is a dictionary containing all the annotations for this
                dataset item, sorted by annotation type.
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
            route=f"slice/{self.slice_id}/exportForTraining",
            requests_command=requests.get,
        )
        return convert_export_payload(api_payload[EXPORTED_ROWS])

    def send_to_labeling(self, project_id: str):
        response = self._client.make_request(
            {}, f"slice/{self.slice_id}/{project_id}/send_to_labeling"
        )
        return AsyncJob.from_json(response, self._client)

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
            route=f"slice/{self.slice_id}/embeddings",
            requests_command=requests.get,
        )
        return api_payload


def check_annotations_are_in_slice(
    annotations: List[Annotation], slice_to_check: Slice
) -> Tuple[bool, Set[str]]:
    """Check membership of the annotation targets within this slice.

    annotations: Annnotations with ids referring to targets.
    slice: The slice to check against.


    Returns:
        A tuple, where the first element is true/false whether the annotations are all
        in the slice.
        The second element is the list of item_ids not in the slice.
        The third element is the list of ref_ids not in the slice.
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
