import copy

import pytest
import requests

from nucleus import BoxAnnotation, Dataset, NucleusClient, Slice
from nucleus.constants import ANNOTATIONS_KEY, BOX_TYPE, ITEM_KEY
from nucleus.job import AsyncJob

from .helpers import (
    TEST_BOX_ANNOTATIONS,
    TEST_PROJECT_ID,
    TEST_SLICE_NAME,
    get_uuid,
)


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    client = NucleusClient(api_key="fake_key")
    test_repr(Slice(slice_id="fake_slice_id", client=client))


def test_slice_create_and_delete_and_list(dataset):
    ds_items = dataset.items

    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[item.reference_id for item in ds_items[:2]],
    )

    dataset_slices = dataset.slices
    assert len(dataset_slices) == 1
    assert slc.id == dataset_slices[0]

    assert slc.name == TEST_SLICE_NAME
    assert slc.dataset_id == dataset.id

    items = slc.items
    assert len(items) == 2
    for item in ds_items[:2]:
        assert (
            item.reference_id == items[0]["ref_id"]
            or item.reference_id == items[1]["ref_id"]
        )

    response = slc.info()
    assert response["name"] == TEST_SLICE_NAME
    assert response["slice_id"] == slc.slice_id
    assert response["dataset_id"] == dataset.id


def test_slice_create_and_export(dataset):
    # Dataset upload
    ds_items = dataset.items

    annotation_in_slice = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[item.reference_id for item in ds_items[:1]],
    )

    dataset.annotate(annotations=[annotation_in_slice])

    expected_box_annotation = copy.deepcopy(annotation_in_slice)

    exported = slc.items_and_annotations()
    assert exported[0][ITEM_KEY] == ds_items[0]
    assert exported[0][ANNOTATIONS_KEY][BOX_TYPE][0] == expected_box_annotation


def test_slice_append(dataset):
    ds_items = dataset.items

    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[ds_items[0].reference_id],
    )

    # Insert duplicate first item
    slc.append(reference_ids=[item.reference_id for item in ds_items[:3]])

    items = slc.items
    assert len(items) == 3
    for item in ds_items[:3]:
        assert (
            item.reference_id == items[0]["ref_id"]
            or item.reference_id == items[1]["ref_id"]
            or item.reference_id == items[2]["ref_id"]
        )

    all_stored_items = [_[ITEM_KEY] for _ in slc.items_and_annotations()]

    def sort_by_reference_id(items):
        # Remove the generated item_ids and standardize
        #  empty metadata so we can do an equality check.
        for item in items:
            item.item_id = None
            if item.metadata == {}:
                item.metadata = None
        return sorted(items, key=lambda x: x.reference_id)

    assert sort_by_reference_id(all_stored_items) == sort_by_reference_id(
        ds_items[:3]
    )


@pytest.mark.skip(reason="404 not found error")
@pytest.mark.integration
def test_slice_send_to_labeling(dataset):
    ds_items = dataset.items

    # Slice creation
    slc = dataset.create_slice(
        name=(TEST_SLICE_NAME + get_uuid()),
        reference_ids=[ds_items[0].reference_id, ds_items[1].reference_id],
    )

    items = slc.items
    assert len(items) == 2

    response = slc.send_to_labeling(TEST_PROJECT_ID)
    assert isinstance(response, AsyncJob)


def test_slice_export_raw_items(dataset: Dataset):
    # Dataset upload
    ds_items = dataset.items
    orig_url = ds_items[0].image_location

    # Slice creation
    slc = dataset.create_slice(
        name=(TEST_SLICE_NAME + "-raw-export"),
        reference_ids=[ds_items[0].reference_id],
    )

    # Export single raw item
    res = slc.export_raw_items()
    export_url = res["raw_dataset_items"][0]["scale_url"]

    orig_bytes = requests.get(orig_url).content
    export_bytes = requests.get(export_url).content

    assert hash(orig_bytes) == hash(export_bytes)


def test_slice_dataset_item_iterator(dataset):
    all_items = dataset.items
    test_slice = dataset.create_slice(
        name=TEST_SLICE_NAME + get_uuid(),
        reference_ids=[item.reference_id for item in all_items[:1]],
    )
    expected_items = {item.reference_id: item for item in test_slice.items}
    actual_items = {
        item.reference_id: item
        for item in test_slice.items_generator(page_size=1)
    }
    for key in actual_items:
        assert actual_items[key] == expected_items[key]
