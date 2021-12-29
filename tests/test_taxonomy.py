import pytest

from .helpers import TEST_DATASET_NAME


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)

    ds.add_taxonomy(
        "[Pytest] taxonomy",
        "category",
        ["[Pytest] taxonomy label 1", "[Pytest] taxonomy label 2"],
    )
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


def test_create_taxonomy(dataset):
    response = dataset.add_taxonomy(
        "New [Pytest] taxonomy",
        "category",
        ["New [Pytest] taxonomy label 1", "New [Pytest] taxonomy label 2"],
    )

    assert response["dataset_id"] == dataset.id
    assert response["taxonomy_name"] == "New [Pytest] taxonomy"
    assert response["status"] == "Taxonomy created"


def test_duplicate_taxonomy(dataset):
    response = dataset.add_taxonomy(
        "[Pytest] taxonomy",
        "category",
        [
            "[Pytest] taxonomy label 1",
            "[Pytest] taxonomy label 2",
            "[Pytest] extra taxonomy label",
        ],
        False,
    )

    assert response["dataset_id"] == dataset.id
    assert response["taxonomy_name"] == "[Pytest] taxonomy"
    assert response["status"] == "Taxonomy already exists"


def test_duplicate_taxonomy_update(dataset):
    response = dataset.add_taxonomy(
        "[Pytest] taxonomy",
        "category",
        [
            "[Pytest] taxonomy label 1",
            "[Pytest] taxonomy label 2",
            "[Pytest] extra taxonomy label",
        ],
        True,
    )

    assert response["dataset_id"] == dataset.id
    assert response["taxonomy_name"] == "[Pytest] taxonomy"
    assert response["status"] == "Taxonomy updated"
