import pytest
from nucleus import Slice, NucleusClient, DatasetItem, BoxAnnotation
from nucleus.constants import ERROR_PAYLOAD, ITEM_KEY
from helpers import (
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    TEST_SLICE_NAME,
    TEST_BOX_ANNOTATIONS,
    reference_id_from_url,
)


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    client = NucleusClient(api_key="fake_key")
    test_repr(Slice(slice_id="fake_slice_id", client=client))


def test_slice_create_and_delete_and_list(dataset):
    # Dataset upload
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
            )
        )
    response = dataset.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()

    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[item.reference_id for item in ds_items[:2]],
    )

    dataset_slices = dataset.slices
    assert len(dataset_slices) == 1
    assert slc.slice_id == dataset_slices[0]

    response = slc.info()
    assert response["name"] == TEST_SLICE_NAME
    assert response["dataset_id"] == dataset.id
    assert len(response["dataset_items"]) == 2
    for item in ds_items[:2]:
        assert (
            item.reference_id == response["dataset_items"][0]["ref_id"]
            or item.reference_id == response["dataset_items"][1]["ref_id"]
        )


def test_slice_create_and_annotate(dataset):
    # Dataset upload
    url = TEST_IMG_URLS[0]
    annotation_in_slice = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    annotation_not_in_slice = BoxAnnotation(**TEST_BOX_ANNOTATIONS[1])

    ds_items = []
    ds_items.append(
        DatasetItem(
            image_location=url,
            reference_id=reference_id_from_url(url),
        )
    )
    response = dataset.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()

    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[item.reference_id for item in ds_items[:2]],
    )

    slc.annotate(annotations=[annotation_in_slice])
    with pytest.raises(ValueError) as not_in_slice_error:
        slc.annotate(annotations=[annotation_not_in_slice])

    assert (
        annotation_not_in_slice.reference_id
        in not_in_slice_error.value.args[0]
    )

    slc.annotate(annotations=[annotation_not_in_slice], strict=False)


def test_slice_append(dataset):
    # Dataset upload
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
            )
        )
    response = dataset.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()

    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[ds_items[0].reference_id],
    )

    # Insert duplicate first item
    slc.append(reference_ids=[item.reference_id for item in ds_items[:3]])

    response = slc.info()
    assert len(response["dataset_items"]) == 3
    for item in ds_items[:3]:
        assert (
            item.reference_id == response["dataset_items"][0]["ref_id"]
            or item.reference_id == response["dataset_items"][1]["ref_id"]
            or item.reference_id == response["dataset_items"][2]["ref_id"]
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
