"""Unit tests for Dataset.get_item_image_bytes (no live API)."""

from unittest.mock import MagicMock, patch

import pytest

from nucleus import NucleusClient
from nucleus.dataset import Dataset
from nucleus.errors import NucleusAPIError

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _dataset():
    client = NucleusClient(api_key="test", endpoint="https://example.test")
    return Dataset("ds_1", client)


def _response(status_code, content=b""):
    response = MagicMock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    response.content = content
    return response


def test_returns_raw_image_bytes_from_loc_route():
    with patch(
        "requests.get", autospec=True, return_value=_response(200, PNG_BYTES)
    ) as get:
        assert _dataset().get_item_image_bytes("di_1") == PNG_BYTES

    url = get.call_args.args[0]
    assert url == "https://example.test/dataset/ds_1/loc/di_1?format=image"


def test_raises_nucleus_api_error_on_failure():
    with patch(
        "requests.get",
        autospec=True,
        return_value=_response(404, b"not found"),
    ):
        with pytest.raises(NucleusAPIError):
            _dataset().get_item_image_bytes("di_missing")
