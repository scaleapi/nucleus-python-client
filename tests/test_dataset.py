import pytest

import nucleus
from nucleus import Dataset


TEST_DATASET_NAME = '[PyTest] Test Dataset'

@pytest.fixture(scope='module')
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
