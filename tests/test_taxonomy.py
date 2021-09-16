import pytest

from .helpers import (
    TEST_DATASET_NAME,
)


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


def test_add_taxonomy(dataset):
    response = dataset.add_taxonomy(
        "[Pytest] taxonomy",
        "category",
        ["[Pytest] taxonomy label 1", "[Pytest] taxonomy label 2"],
    )

    assert response["dataset_id"] == dataset.id
    assert response["taxonomy_name"] == "[Pytest] taxonomy"
    assert response["type"] == "category"
