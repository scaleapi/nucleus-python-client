class Slice:
    """
    Slice respesents a subset of your Dataset.
    """

    def __init__(self, slice_id: str, client):
        self.slice_id = slice_id
        self._client = client

    @property
    def info(self) -> dict:
        """
        This endpoint provides information about specified slice.

        :param
        slice_id: id of the slice
        id_type: the type of IDs you want in response (either "reference_id" or "dataset_item_id")
        to identify the DatasetItems

        :return:
        {
            "name": str,
            "dataset_id": str,
            "dataset_item_ids": List[str],
        }
        """
        return self._client.slice_info(self.slice_id)
