import pytest

from pathlib import Path

from nucleus import Dataset, DatasetItem, UploadResponse
from nucleus.constants import (
    NEW_ITEMS,
    UPDATED_ITEMS,
    IGNORED_ITEMS,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    DATASET_ID_KEY,
)


TEST_DATASET_NAME = '[PyTest] Test Dataset'
TEST_SLICE_NAME = '[PyTest] Test Slice'
TEST_IMG_URLS = [
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/6dd63871-831611a6.jpg',
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/82c1005c-e2d1d94f.jpg',
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/7f2e1814-6591087d.jpg',
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/06924f46-1708b96f.jpg',
    's3://scaleapi-attachments/BDD/BDD/bdd100k/images/100k/train/89b42832-10d662f4.jpg',
]

@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    yield ds

    CLIENT.delete_dataset(ds.id)

def test_dataset_create_and_delete(CLIENT):
    # Creation
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    assert isinstance(ds, Dataset)
    assert ds.name == TEST_DATASET_NAME
    assert ds.model_runs == []
    assert ds.slices == []
    assert ds.size == 0
    assert ds.items == []

    # Deletion
    response = CLIENT.delete_dataset(ds.id)
    assert response == {}


def test_dataset_append(dataset):
    def check_is_expected_response(response):
        assert isinstance(response, UploadResponse)
        resp_json = response.json()
        assert resp_json[DATASET_ID_KEY] == dataset.id
        assert resp_json[NEW_ITEMS] == len(TEST_IMG_URLS)
        assert resp_json[UPDATED_ITEMS] == 0
        assert resp_json[IGNORED_ITEMS] == 0
        assert resp_json[ERROR_ITEMS] == 0
        assert ERROR_PAYLOAD not in resp_json

    # Plain image upload
    ds_items_plain = []
    for url in TEST_IMG_URLS:
        ds_items_plain.append(DatasetItem(image_location=url))
    response = dataset.append(ds_items_plain)
    check_is_expected_response(response)

    # With reference ids and metadata:
    ds_items_with_metadata = []
    for i, url in enumerate(TEST_IMG_URLS):
        ds_items_with_metadata.append(DatasetItem(
            image_location=url,
            reference_id=Path(url).name,
            metadata={
                'made_with_pytest': True,
                'example_int': i,
                'example_str': 'hello',
                'example_float': 0.5,
                'example_dict': {
                    'nested': True,
                },
                'example_list': ['hello', i, False],
            }
        ))

    response = dataset.append(ds_items_with_metadata)
    check_is_expected_response(response)


def test_slice_create_and_delete(dataset):
    # Dataset upload
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(DatasetItem(
            image_location=url,
            reference_id=Path(url).name,
        ))
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

    response = slc.info(id_type="reference_id")
    assert response["name"] == TEST_SLICE_NAME
    assert response["dataset_id"] == dataset.id
    assert len(response["reference_ids"]) == 2
    for item in ds_items[:2]:
        assert item.reference_id in response["reference_ids"]
    print(response)


def test_slice_append(dataset):
    # Dataset upload
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(DatasetItem(
            image_location=url,
            reference_id=Path(url).name,
        ))
    response = dataset.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()

    # Slice creation
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[ds_items[0].reference_id],
    )

    # Insert duplicate first item
    slc.append(reference_ids=[item.reference_id for item in ds_items[:3]])

    response = slc.info(id_type="reference_ids")
    assert len(response["reference_ids"]) == 3
    for item in ds_items[:3]:
        assert item.reference_id in response["reference_ids"]
