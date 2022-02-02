from copy import deepcopy
from enum import Enum
from typing import List, TYPE_CHECKING, Union

import requests

from nucleus.pydantic_base import DictCompatibleModel
from .utils import ListOrStringOrNone, enforce_list

if TYPE_CHECKING:
    from . import NucleusClient


# Wording set to match with backend enum
class ExportMetadataType(Enum):
    SCENES = "scene"
    DATASET_ITEMS = "item"


def contains_initial_keys(dict_a, dict_b):
    a_keys = set(dict_a.keys())
    b_keys = set(dict_b.keys())
    diff = b_keys.difference(a_keys)
    if len(diff) > 0:
        # TODO(Jean): raise Exception here once we support overwrite/replace
        print(
            f"Detected key deletion of key(s): {diff}.\n"
            f"Key deletion is not supported at the moment. Only key addition or updates."
        )


class DatasetItemMetadata(DictCompatibleModel):
    reference_id: str
    metadata: dict

    def to_dict(self):
        return {"reference_id": self.reference_id, "metadata": self.metadata}


class SceneMetadata(DictCompatibleModel):
    reference_id: str
    metadata: dict
    dataset_items: List[DatasetItemMetadata]

    def to_dict(self):
        return {
            "reference_id": self.reference_id,
            "metadata": self.metadata,
            "dataset_items": [item.to_dict() for item in self.dataset_items],
        }


SceneOrItemMetadataList = Union[List[SceneMetadata], List[DatasetItemMetadata]]


class MetadataManager:
    def __init__(self, dataset_id: str, client: "NucleusClient"):
        self.dataset_id = dataset_id
        self._client = client

        # for internal validation
        self._raw_items_flat_map = {}  # ref_id => metadata

    def _populate_flat_map(self, data: SceneOrItemMetadataList):
        data_copy = deepcopy(data)
        for item in data_copy:
            self._raw_items_flat_map[item.reference_id] = item.metadata
            if isinstance(item, SceneMetadata):
                for dataset_item in item.dataset_items:
                    self._raw_items_flat_map[
                        dataset_item.reference_id
                    ] = dataset_item.metadata

    def validate_scene_items(self, new_items: SceneOrItemMetadataList):
        # assert that all items are of the same type
        assert all(
            isinstance(item, SceneMetadata) for item in new_items
        ) or all(
            isinstance(item, DatasetItemMetadata) for item in new_items
        ), "When updating items, the list must be of all the same type. Either 'SceneMetadata' or 'DatasetItemMetadata'"

        # If user deletes a key, print warning message
        for item in new_items:
            contains_initial_keys(
                item.metadata, self._raw_items_flat_map[item.reference_id]
            )
            if isinstance(item, SceneMetadata):
                for dataset_item in item.dataset_items:
                    contains_initial_keys(
                        dataset_item.metadata,
                        self._raw_items_flat_map[dataset_item.reference_id],
                    )

    def load_scenes(
        self, scene_ref_ids: ListOrStringOrNone, with_items: bool = False
    ) -> List[SceneMetadata]:
        scene_ref_ids = enforce_list(scene_ref_ids)

        payload = {"ref_ids": scene_ref_ids, "with_items": with_items}
        resp = self._client.make_request(
            payload=payload,
            requests_command=requests.get,
            route=f"dataset/{self.dataset_id}/metadata_by_level/{ExportMetadataType.SCENES.value}",
        )

        scene_items = [
            SceneMetadata.parse_obj(resp_item)
            for resp_item in resp["scenes_metadata"]
        ]
        self._populate_flat_map(scene_items)
        return scene_items

    def load_dataset_items(
        self, item_ref_ids: ListOrStringOrNone
    ) -> List[DatasetItemMetadata]:
        item_ref_ids = enforce_list(item_ref_ids)

        payload = {"ref_ids": item_ref_ids}
        resp = self._client.make_request(
            payload=payload,
            requests_command=requests.get,
            route=f"dataset/{self.dataset_id}/metadata_by_level/{ExportMetadataType.DATASET_ITEMS.value}",
        )
        dataset_items = [
            DatasetItemMetadata.parse_obj(resp_item)
            for resp_item in resp["items_metadata"]
        ]
        self._populate_flat_map(dataset_items)
        return dataset_items

    def update(self, items: SceneOrItemMetadataList):
        # TODO(Jean): Support overwrite/replace in the future
        self.validate_scene_items(items)

        if isinstance(items, (SceneMetadata, DatasetItemMetadata)):
            items = [items]

        dict_items = [item.to_dict() for item in items]
        level = ExportMetadataType.DATASET_ITEMS.value
        if isinstance(items[0], SceneMetadata):
            level = ExportMetadataType.SCENES.value

        payload = {"metadata": dict_items, "level": level}
        resp = self._client.make_request(
            payload=payload, route=f"dataset/{self.dataset_id}/metadata"
        )
        return resp
