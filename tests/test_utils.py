import io

import pytest

from nucleus import (
    BoxAnnotation,
    BoxPrediction,
    CategoryAnnotation,
    CategoryPrediction,
    DatasetItem,
    LineAnnotation,
    LinePrediction,
    Point,
    PolygonAnnotation,
    PolygonPrediction,
    utils,
)


class TestNonSerializableObject:
    def weird_function():
        print("can't touch this. Dun dun dun dun.")


def test_serialize():
    test_items = [
        DatasetItem("fake_url1", "fake_id1"),
        DatasetItem(
            "fake_url2",
            "fake_id2",
            metadata={
                "ok": "field",
                "bad": TestNonSerializableObject(),
            },
        ),
    ]

    with io.StringIO() as in_memory_filelike:
        with pytest.raises(ValueError) as error:
            utils.serialize_and_write(
                test_items,
                in_memory_filelike,
            )
        assert "DatasetItem" in str(error.value)
        assert "fake_id2" in str(error.value)
        assert "fake_id1" not in str(error.value)

        test_items[1].metadata["bad"] = "fixed"

        utils.serialize_and_write(test_items, in_memory_filelike)


# --------------------------------------------------------------------------- #
# convert_export_payload: dataset_item_id propagation (offline, no live API)
# --------------------------------------------------------------------------- #
REF_ID = "ref_1"
DI_ID = "di_abc123"


def _annotation_payloads():
    """Minimal but valid per-type payloads, keyed by the export-row type key.

    Built by serializing real annotation objects so the shapes stay in sync
    with ``from_json`` — reference_id/dataset_item_id are omitted here because
    the backend returns them on the item, not per object.
    """
    verts = [Point(0, 0), Point(1, 0), Point(1, 1)]
    return {
        "box": BoxAnnotation("car", 0, 0, 4, 4, REF_ID).to_payload(),
        "polygon": PolygonAnnotation("car", verts, REF_ID).to_payload(),
        "line": LineAnnotation("lane", verts, REF_ID).to_payload(),
        "category": CategoryAnnotation("car", "vehicles", REF_ID).to_payload(),
    }


def _export_row(dataset_item_id=DI_ID, with_objects=True):
    """One row shaped like the batch-export endpoint returns."""
    payloads = _annotation_payloads() if with_objects else {}
    item = {"url": "https://example.com/a.jpg", "reference_id": REF_ID}
    if dataset_item_id is not None:
        item["dataset_item_id"] = dataset_item_id
    return {
        "item": item,
        "annotations": [],
        "segmentation": None,
        "box": [payloads["box"]] if with_objects else [],
        "polygon": [payloads["polygon"]] if with_objects else [],
        "line": [payloads["line"]] if with_objects else [],
        "keypoints": [],
        "cuboid": [],
        "category": [payloads["category"]] if with_objects else [],
        "multicategory": [],
    }


@pytest.mark.parametrize("obj_type", ["box", "polygon", "line", "category"])
def test_convert_export_stamps_dataset_item_id_on_annotations(obj_type):
    result = utils.convert_export_payload([_export_row()])
    item = result[0]["item"]
    obj = result[0]["annotations"][obj_type][0]
    assert item.dataset_item_id == DI_ID
    assert obj.dataset_item_id == DI_ID
    assert obj.reference_id == REF_ID


@pytest.mark.parametrize("obj_type", ["box", "polygon", "line", "category"])
def test_convert_export_stamps_dataset_item_id_on_predictions(obj_type):
    result = utils.convert_export_payload(
        [_export_row()], has_predictions=True
    )
    obj = result[0]["predictions"][obj_type][0]
    assert obj.dataset_item_id == DI_ID
    assert obj.reference_id == REF_ID


def test_convert_export_leaves_dataset_item_id_none_when_backend_omits_it():
    """An un-upgraded backend that omits the id must not raise."""
    result = utils.convert_export_payload([_export_row(dataset_item_id=None)])
    assert result[0]["item"].dataset_item_id is None
    for obj_type in ("box", "polygon", "line", "category"):
        assert result[0]["annotations"][obj_type][0].dataset_item_id is None


def test_dataset_item_id_is_read_only_on_annotations():
    """Server-assigned: excluded from __eq__ and never sent in to_payload."""
    exported = utils.convert_export_payload([_export_row()])[0]["annotations"][
        "box"
    ][0]
    local = BoxAnnotation("car", 0, 0, 4, 4, REF_ID)

    # Equal despite the exported copy carrying an id the local one lacks.
    assert local.dataset_item_id is None
    assert exported.dataset_item_id == DI_ID
    assert exported == local
    # The id is not part of the upload contract.
    assert "dataset_item_id" not in exported.to_payload()


def test_dataset_item_id_populates_from_dataset_item_json():
    item = DatasetItem.from_json(
        {
            "url": "https://example.com/a.jpg",
            "reference_id": REF_ID,
            "dataset_item_id": DI_ID,
        }
    )
    assert item.dataset_item_id == DI_ID
    assert "dataset_item_id" not in item.to_payload()


def test_format_dataset_item_response_stamps_dataset_item_id():
    """Single-item loc/refloc/iloc annotations carry the id like exports do."""
    response = {
        "item": {
            "url": "https://example.com/a.jpg",
            "reference_id": REF_ID,
            "dataset_item_id": DI_ID,
        },
        "annotations": {
            "box": [BoxAnnotation("car", 0, 0, 4, 4, REF_ID).to_payload()]
        },
    }
    out = utils.format_dataset_item_response(response)
    assert out["item"].dataset_item_id == DI_ID
    assert out["annotations"]["box"][0].dataset_item_id == DI_ID


def test_format_dataset_item_response_none_when_backend_omits_id():
    response = {
        "item": {"url": "https://example.com/a.jpg", "reference_id": REF_ID},
        "annotations": {
            "box": [BoxAnnotation("car", 0, 0, 4, 4, REF_ID).to_payload()]
        },
    }
    out = utils.format_dataset_item_response(response)
    assert out["annotations"]["box"][0].dataset_item_id is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda di: BoxAnnotation(
            "car", 0, 0, 4, 4, REF_ID, dataset_item_id=di
        ),
        lambda di: BoxPrediction(
            "car", 0, 0, 4, 4, REF_ID, dataset_item_id=di
        ),
    ],
)
def test_dataset_item_id_is_keyword_only(factory):
    """Server-assigned, so it must be passed by keyword, never positionally."""
    # Keyword works.
    assert factory("di_kw").dataset_item_id == "di_kw"


def test_dataset_item_id_rejected_positionally():
    """A positional value would be silently dropped by to_payload — block it."""
    with pytest.raises(TypeError):
        # One arg past track_reference_id would land on the old positional slot.
        BoxPrediction(
            "car", 0, 0, 4, 4, REF_ID, 0.9, "a1", {}, {}, None, None, "di_x"
        )
