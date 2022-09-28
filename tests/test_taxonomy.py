import pytest

from .helpers import TEST_DATASET_NAME


@pytest.fixture()
def taxonomy_dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME, is_scene=False)

    ds.add_taxonomy(
        "[Pytest] taxonomy",
        "category",
        ["[Pytest] taxonomy label 1", "[Pytest] taxonomy label 2"],
    )
    yield ds


def test_create_taxonomy(taxonomy_dataset):
    response = taxonomy_dataset.add_taxonomy(
        "New [Pytest] taxonomy",
        "category",
        ["New [Pytest] taxonomy label 1", "New [Pytest] taxonomy label 2"],
    )

    assert response["dataset_id"] == taxonomy_dataset.id
    assert response["taxonomy_name"] == "New [Pytest] taxonomy"
    assert response["status"] == "Taxonomy created"


def test_duplicate_taxonomy(taxonomy_dataset):
    response = taxonomy_dataset.add_taxonomy(
        "[Pytest] taxonomy",
        "category",
        [
            "[Pytest] taxonomy label 1",
            "[Pytest] taxonomy label 2",
            "[Pytest] extra taxonomy label",
        ],
        False,
    )

    assert response["dataset_id"] == taxonomy_dataset.id
    assert response["taxonomy_name"] == "[Pytest] taxonomy"
    assert response["status"] == "Taxonomy already exists"


def test_duplicate_taxonomy_update(taxonomy_dataset):
    response = taxonomy_dataset.add_taxonomy(
        "[Pytest] taxonomy",
        "category",
        [
            "[Pytest] taxonomy label 1",
            "[Pytest] taxonomy label 2",
            "[Pytest] extra taxonomy label",
        ],
        True,
    )

    assert response["dataset_id"] == taxonomy_dataset.id
    assert response["taxonomy_name"] == "[Pytest] taxonomy"
    assert response["status"] == "Taxonomy updated"


def test_delete_taxonomy(taxonomy_dataset):
    response = taxonomy_dataset.delete_taxonomy(
        "[Pytest] taxonomy",
    )

    assert response["dataset_id"] == taxonomy_dataset.id
    assert response["taxonomy_name"] == "[Pytest] taxonomy"
    assert response["status"] == "Taxonomy successfully deleted"
