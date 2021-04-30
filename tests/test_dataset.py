import pytest
import os

from helpers import (
    TEST_SLICE_NAME,
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    LOCAL_FILENAME,
    reference_id_from_url,
)

from nucleus import (
    Dataset,
    DatasetItem,
    UploadResponse,
    NucleusClient,
    NucleusAPIError,
)
from nucleus.constants import (
    NEW_ITEMS,
    UPDATED_ITEMS,
    IGNORED_ITEMS,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    DATASET_ID_KEY,
)

TEST_AUTOTAG_DATASET = "ds_bz43jm2jwm70060b3890"


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    test_repr(
        DatasetItem(
            image_location="test_url",
            reference_id="test_reference_id",
            metadata={
                "made_with_pytest": True,
                "example_int": 0,
                "example_str": "hello",
                "example_float": 0.5,
                "example_dict": {
                    "nested": True,
                },
                "example_list": ["hello", 1, False],
            },
        )
    )
    test_repr(Dataset("test_dataset", NucleusClient(api_key="fake_key")))


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


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
    assert response == {"message": "Beginning dataset deletion..."}


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
        ds_items_with_metadata.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
                metadata={
                    "made_with_pytest": True,
                    "example_int": i,
                    "example_str": "hello",
                    "example_float": 0.5,
                    "example_dict": {
                        "nested": True,
                    },
                    "example_list": ["hello", i, False],
                },
            )
        )

    response = dataset.append(ds_items_with_metadata)
    check_is_expected_response(response)


def test_dataset_append_local(CLIENT, dataset):
    ds_items_local = [DatasetItem(image_location=LOCAL_FILENAME)]
    response = dataset.append(ds_items_local)
    assert isinstance(response, UploadResponse)
    resp_json = response.json()
    assert resp_json[DATASET_ID_KEY] == dataset.id
    assert resp_json[NEW_ITEMS] == 1
    assert resp_json[UPDATED_ITEMS] == 0
    assert resp_json[IGNORED_ITEMS] == 0
    assert resp_json[ERROR_ITEMS] == 0
    assert ERROR_PAYLOAD not in resp_json


def test_dataset_list_autotags(CLIENT, dataset):
    # Creation
    # List of Autotags should be empty
    autotag_response = CLIENT.list_autotags(dataset.id)
    assert autotag_response == []


def test_raises_error_for_duplicate():
    fake_dataset = Dataset("fake", NucleusClient("fake"))
    with pytest.raises(ValueError) as error:
        fake_dataset.append(
            [
                DatasetItem("fake", "duplicate"),
                DatasetItem("fake", "duplicate"),
            ]
        )
    assert (
        str(error.value)
        == "Duplicate reference ids found among dataset_items:"
        " {'duplicate': 'Count: 2'}"
    )


def test_dataset_export_autotag_scores(CLIENT):
    # This test can only run for the test user who has an indexed dataset.
    # TODO: if/when we can create autotags via api, create one instead.
    if os.environ.get("HAS_ACCESS_TO_TEST_DATA", False):
        dataset = CLIENT.get_dataset(TEST_AUTOTAG_DATASET)

        with pytest.raises(NucleusAPIError) as api_error:
            dataset.autotag_scores(autotag_name="NONSENSE_GARBAGE")
        assert (
            f"The autotag NONSENSE_GARBAGE was not found in dataset {TEST_AUTOTAG_DATASET}"
            in str(api_error.value)
        )

        scores = dataset.autotag_scores(autotag_name="TestTag")

        for column in ["dataset_item_ids", "ref_ids", "scores"]:
            assert column in scores
            assert len(scores[column]) > 0
