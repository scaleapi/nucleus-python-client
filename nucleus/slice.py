from typing import List


class Slice:
    """
    Slice respesents a subset of your Dataset.
    """

    def __init__(self, slice_id: str, client):
        self.slice_id = slice_id
        self._client = client

    def __repr__(self):
        return f"Slice(slice_id='{self.slice_id}', client={self._client})"

    def __eq__(self, other):
        if self.slice_id == other.slice_id:
            if self._client == other._client:
                return True
        return False

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
        response = self._client.append_to_slice(
            slice_id=self.slice_id,
            dataset_item_ids=dataset_item_ids,
            reference_ids=reference_ids,
        )
        return response
