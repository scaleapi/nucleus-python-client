import pytest
from nucleus import (
    Slice,
    NucleusClient
)



def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object
    client = NucleusClient(api_key="fake_key")
    test_repr(Slice(slice_id="fake_slice_id", client=client))