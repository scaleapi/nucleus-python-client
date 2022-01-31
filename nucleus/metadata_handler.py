from enum import Enum
from typing import List, Optional, TYPE_CHECKING, Union

import requests

from nucleus.pydantic_base import DictCompatibleModel

if TYPE_CHECKING:
    from . import NucleusClient



class ExportMetadataType(Enum):
    SCENES = "scenes"
    DATASET_ITEMS = "datasetItems"


def same_keys(dict_a, dict_b):
    a_keys = sorted(dict_a.keys())
    b_keys = sorted(dict_b.keys())
    if (a_keys != b_keys):
        raise Exception(
            f"Detected key deletion on metadata. Previous: {dict_b}. New: {dict_a}."
            f"If this is desired, set overwrite=True when calling the 'update' method."
        )


class DatasetItemMetadata(DictCompatibleModel):
    reference_id: str
    metadata: dict


class SceneMetadata(DictCompatibleModel):
    reference_id: str
    metadata: dict
    dataset_items: List[DatasetItemMetadata]


SceneOrItemMetadata = Union[List[SceneMetadata], List[DatasetItemMetadata]]
ListOrString = Optional[Union[List[str], str]]


def validate_scene_items(new_items: SceneOrItemMetadata, old_items_map: dict):
    # If user deletes a key, and does not set overwrite=True, throw error
    for item in new_items:
        same_keys(item.metadata, old_items_map[item.reference_id])
        if isinstance(item, SceneMetadata):
            for dataset_item in item.dataset_items:
                same_keys(dataset_item.metadata, old_items_map[dataset_item.reference_id])


def enforce_list(items: ListOrString):
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    return items


class MetadataManager:

    def __init__(self, dataset_id: str, client: "NucleusClient"):
        self.dataset_id = dataset_id
        self._client = client

        # for internal validation
        self._raw_items_flat_map = {}  # ref_id => metadata

    def _populate_flat_map(self, data: SceneOrItemMetadata):
        for item in data:
            self._raw_items_flat_map[item.reference_id] = item.metadata
            if isinstance(item, SceneMetadata):
                for dataset_item in item.dataset_items:
                    self._raw_items_flat_map[dataset_item.reference_id] = dataset_item.metadata

    def load_scenes(self, scene_ref_ids: ListOrString = None, with_items: bool = False) -> List[SceneMetadata]:
        scene_ref_ids = enforce_list(scene_ref_ids)

        payload = {'ref_ids': scene_ref_ids, 'with_items': with_items}
        resp = self._client.make_request(
            payload=payload,
            requests_command=requests.get,
            route=f"dataset/{self.dataset_id}/metadata_by_type/{ExportMetadataType.SCENES.value}",
        )

        scene_items = [SceneMetadata.parse_obj(resp_item) for resp_item in resp["scenes_metadata"]]
        self._populate_flat_map(scene_items)
        return scene_items

    def load_dataset_items(self, item_ref_ids: ListOrString) -> List[DatasetItemMetadata]:
        item_ref_ids = enforce_list(item_ref_ids)

        payload = {'ref_ids': item_ref_ids}
        resp = self._client.make_request(
            payload=payload,
            requests_command=requests.get,
            route=f"dataset/{self.dataset_id}/metadata_by_type/{ExportMetadataType.DATASET_ITEMS.value}",
        )
        dataset_items = [DatasetItemMetadata.parse_obj(resp_item) for resp_item in resp["items_metadata"]]
        self._populate_flat_map(dataset_items)
        return dataset_items

    def update(self, items: SceneOrItemMetadata, overwrite: bool = False):
        if (overwrite):
            # TODO(Jean): Should we request user confirmation for this? Or put a timeout so they can cancel?
            print("Overwrite flag set to True, original metadata will be deleted, and replaced by the new one.")
        else:
            validate_scene_items(items, self._raw_items_flat_map)

        return
        payload = {"metadata": items, "overwrite": overwrite, "is_scene": isinstance(items, SceneMetadata)}
        resp = self._client.make_request(payload=payload, route=f"dataset/{self.dataset_id}/metadata")
        return resp

    def update_raw(self, from_dict: dict = None, from_json: str = None, overwrite: bool = False):
        raise NotImplementedError

