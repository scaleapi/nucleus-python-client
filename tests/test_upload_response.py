import pytest
from nucleus import Slice, UploadResponse


from nucleus.constants import (
    DATASET_ID_KEY,
)


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    test_repr(UploadResponse(json={DATASET_ID_KEY: "fake_dataset_id_key"}))
