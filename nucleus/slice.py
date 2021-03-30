from typing import List, Dict, Any
from .constants import (
    DATASET_ITEM_ID_KEY,
    REFERENCE_IDS_KEY,
    DATASET_ITEM_IDS_KEY,
)


class Slice:
    """
    Slice respesents a subset of your Dataset.
    """

    def __init__(self, slice_id: str, client):
        self.slice_id = slice_id
        self._client = client

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
        return self._client.slice_info(self.slice_id)

    def append(
        self,
        dataset_item_ids: List[str] = None,
        reference_ids: List[str] = None,
    ) -> dict:
        """
        Appends to a slice from items already present in a dataset.
        The caller must exclusively use either datasetItemIds or reference_ids
        as a means of identifying items in the dataset.

        :param
        dataset_item_ids: List[str],
        reference_ids: List[str],

        :return:
        {
            "slice_id": str,
        }
        """
        if dataset_item_ids and reference_ids:
            raise Exception(
                "You cannot specify both dataset_item_ids and reference_ids"
            )

        payload: Dict[str, Any] = {}
        if dataset_item_ids:
            payload[DATASET_ITEM_IDS_KEY] = dataset_item_ids
        if reference_ids:
            payload[REFERENCE_IDS_KEY] = reference_ids

        response = self._client._make_request(
            payload, f"slice/{self.slice_id}/append"
        )
        return response
