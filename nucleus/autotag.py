from typing import List, Dict, Any
from .constants import (
    DATASET_ITEM_ID_KEY,
    REFERENCE_IDS_KEY,
    DATASET_ITEM_IDS_KEY,
)


class Autotag:
    """
    Autotag respesents a subset of your Dataset tagged by a binary attribute trained through the Nucleus Dashboard.
    """

    def __init__(self, autotag_id: str, client):
        self.autotag_id = autotag_id
        self._client = client

    def info(self) -> dict:
        """
        This endpoint provides information about specified autotag.

        :return:
        {
            "name": str,
            "autotag_id": str,
            "dataset_id": str,
            "dataset_item_ids" or "reference_ids": List[str],
        }
        """
        return self._client.get_autotag(self.autotag_id)
