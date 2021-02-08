import pytest

import nucleus
from nucleus import Dataset


TEST_DATASET_NAME = '[PyTest] Test Dataset'

def test_dataset_create_and_delete(CLIENT):
    # Creation
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    assert isinstance(ds, Dataset)

    # Deletion
    response = CLIENT.delete_dataset(ds.id)
