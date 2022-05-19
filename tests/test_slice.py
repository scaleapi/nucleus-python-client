import copy

import pytest
import requests

from nucleus import BoxAnnotation, BoxPrediction, Dataset, NucleusClient, Slice
from nucleus.constants import (
    ANNOTATIONS_KEY,
    BOX_TYPE,
    ITEM_KEY,
    PREDICTIONS_KEY,
)
from nucleus.job import AsyncJob

from .helpers import (
    TEST_BOX_ANNOTATIONS,
    TEST_BOX_PREDICTIONS,
    TEST_PROJECT_ID,
    TEST_SLICE_NAME,
    get_uuid,
)


@pytest.fixture()
def slc(CLIENT, dataset):
    slice_ref_ids = [item.reference_id for item in dataset.items[:1]]
    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=slice_ref_ids,
    )

    yield slc

    CLIENT.delete_slice(slc.id)


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    client = NucleusClient(api_key="fake_key")
    test_repr(Slice(slice_id="fake_slice_id", client=client))


def test_slice_create_and_delete_and_list(dataset: Dataset):
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

    assert {item.reference_id for item in slc.items} == {
        item.reference_id for item in ds_items[:2]
    }

    response = slc.info()
    assert response["name"] == TEST_SLICE_NAME
    assert response["slice_id"] == slc.slice_id
    assert response["dataset_id"] == dataset.id


def test_slice_create_and_export(dataset):
    # Dataset upload
    ds_items = dataset.items

    slice_ref_ids = [item.reference_id for item in ds_items[:1]]
    # This test assumes one box annotation per item.
    annotations = [
        BoxAnnotation.from_json(json_data)
        for json_data in TEST_BOX_ANNOTATIONS
    ]
    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=slice_ref_ids,
    )

    dataset.annotate(annotations=annotations)

    def get_expected_box_annotation(reference_id):
        for annotation in annotations:
            if annotation.reference_id == reference_id:
                return annotation

    def get_expected_item(reference_id):
        if reference_id not in slice_ref_ids:
            raise ValueError("Got results outside the slice")
        for item in ds_items:
            if item.reference_id == reference_id:
                return item

    exported = slc.items_and_annotations()
    for row in exported:
        reference_id = row[ITEM_KEY].reference_id
        assert row[ITEM_KEY] == get_expected_item(reference_id)
        assert row[ANNOTATIONS_KEY][BOX_TYPE][
            0
        ] == get_expected_box_annotation(reference_id)


def test_slice_create_and_prediction_export(dataset, slc, model):
    # Dataset upload
    ds_items = dataset.items

    predictions = [
        BoxPrediction(**pred_raw) for pred_raw in TEST_BOX_PREDICTIONS
    ]
    response = dataset.upload_predictions(model, predictions)

    assert response

    slice_reference_ids = [item.reference_id for item in slc.items]

    def get_expected_box_prediction(reference_id):
        for prediction in predictions:
            if prediction.reference_id == reference_id:
                return prediction

    def get_expected_item(reference_id):
        if reference_id not in slice_reference_ids:
            raise ValueError("Got results outside the slice")
        for item in ds_items:
            if item.reference_id == reference_id:
                return item

    exported = slc.export_predictions(model)
    for row in exported:
        reference_id = row[ITEM_KEY].reference_id
        assert row[ITEM_KEY] == get_expected_item(reference_id)
        assert row[PREDICTIONS_KEY][BOX_TYPE][
            0
        ] == get_expected_box_prediction(reference_id)


def test_slice_append(dataset):
    ds_items = dataset.items

    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[ds_items[0].reference_id],
    )

    # Insert duplicate first item
    slc.append(reference_ids=[item.reference_id for item in ds_items[:3]])
    slice_items = slc.items

    assert len(slice_items) == 3

    assert {_.reference_id for _ in ds_items[:3]} == {
        _.reference_id for _ in slice_items
    }


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
