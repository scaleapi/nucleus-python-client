import io

import pytest

from nucleus import DatasetItem, utils


class TestNonSerializableObject:
    def weird_function():
        print("can't touch this. Dun dun dun dun.")


def test_serialize():
    test_items = [
        DatasetItem("fake_url1", "fake_id1"),
        DatasetItem(
            "fake_url2",
            "fake_id2",
            metadata={
                "ok": "field",
                "bad": TestNonSerializableObject(),
            },
        ),
    ]

    with io.StringIO() as in_memory_filelike:
        with pytest.raises(ValueError) as error:
            utils.serialize_and_write(
                test_items,
                in_memory_filelike,
            )
        assert "DatasetItem" in str(error.value)
        assert "fake_id2" in str(error.value)
        assert "fake_id1" not in str(error.value)

        test_items[1].metadata["bad"] = "fixed"

        utils.serialize_and_write(test_items, in_memory_filelike)
