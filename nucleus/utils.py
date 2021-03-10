from typing import List, Union, Dict

from .dataset_item import DatasetItem
from .prediction import BoxPrediction, PolygonPrediction


def _get_all_field_values(metadata_list: List[dict], key: str):
    return {metadata[key] for metadata in metadata_list if key in metadata}


def suggest_metadata_schema(
    data: Union[
        List[DatasetItem], List[BoxPrediction], List[PolygonPrediction]
    ]
):
    metadata_list: List[dict] = [
        d.metadata for d in data if d.metadata is not None
    ]
    schema = {}
    all_keys = {k for metadata in metadata_list for k in metadata.keys()}

    all_key_values: Dict[str, set] = {
        k: _get_all_field_values(metadata_list, k) for k in all_keys
    }

    for key, values in all_key_values.items():
        entry: dict = {}
        if all(isinstance(x, (float, int)) for x in values):
            entry["type"] = "number"
        elif len(values) <= 50:
            entry["type"] = "category"
            entry["choices"] = list(values)
        else:
            entry["type"] = "text"
        schema[key] = entry
    return schema
