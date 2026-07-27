"""Unit tests for model weights upload/download (no live API, no real S3)."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from nucleus import Model, ModelWeights, NucleusClient
from nucleus.model_weights import (
    MODEL_WEIGHTS_MAX_BYTES,
    finalize_payload,
    presign_payload,
    stream_weights_to_file,
    transfer_weights_to_storage,
)

_WEIGHTS_DTO = {
    "modelProjectId": "prj_1",
    "present": True,
    "status": "ready",
    "sizeBytes": 2048,
    "originalFilename": "weights.bin",
    "contentType": "application/octet-stream",
    "downloadUrl": "https://s3.example/signed-get",
}


def _client() -> NucleusClient:
    return NucleusClient(api_key="fake_key")


def _model(client) -> Model:
    return Model("prj_1", "My CNN", "My-CNN", {}, client)


# --------------------------------------------------------------------------- #
# DTO parsing
# --------------------------------------------------------------------------- #
def test_model_weights_from_json_maps_camel_case():
    weights = ModelWeights.from_json(_WEIGHTS_DTO)
    assert weights.model_project_id == "prj_1"
    assert weights.present is True
    assert weights.status == "ready"
    assert weights.size_bytes == 2048
    assert weights.original_filename == "weights.bin"
    assert weights.content_type == "application/octet-stream"
    assert weights.download_url == "https://s3.example/signed-get"


def test_model_weights_from_json_absent_artifact():
    weights = ModelWeights.from_json(
        {"modelProjectId": "prj_1", "present": False, "status": None}
    )
    assert weights.present is False
    assert weights.size_bytes is None
    assert weights.download_url is None


# --------------------------------------------------------------------------- #
# Payload builders
# --------------------------------------------------------------------------- #
def test_presign_payload_omits_unset_optionals():
    assert presign_payload(10, None, None, None) == {"declaredSizeBytes": 10}


def test_presign_payload_includes_provided_optionals():
    assert presign_payload(10, "application/zip", "w.bin", "abc123") == {
        "declaredSizeBytes": 10,
        "contentType": "application/zip",
        "originalFilename": "w.bin",
        "checksumSha256": "abc123",
    }


def test_finalize_payload_omits_empty_parts():
    assert finalize_payload("up_1", None) == {"uploadId": "up_1"}
    assert finalize_payload("up_1", []) == {"uploadId": "up_1"}


def test_finalize_payload_includes_parts():
    parts = [{"partNumber": 1, "eTag": "aaa"}]
    assert finalize_payload("up_1", parts) == {
        "uploadId": "up_1",
        "parts": parts,
    }


# --------------------------------------------------------------------------- #
# Storage transfer
# --------------------------------------------------------------------------- #
def _ok_put(etag='"abc"'):
    response = MagicMock()
    response.ok = True
    response.headers = {"ETag": etag}
    return response


def test_transfer_single_put_sends_required_headers(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"x" * 32)
    presign = {
        "uploadId": "up_1",
        "uploadUrl": "https://s3.example/put",
        "requiredHeaders": {"Content-Type": "application/octet-stream"},
        "parts": None,
        "partSizeBytes": None,
    }
    with patch("nucleus.model_weights.requests.put") as mock_put:
        mock_put.return_value = _ok_put()
        parts = transfer_weights_to_storage(str(path), presign, 32)

    assert parts is None
    assert mock_put.call_count == 1
    _, kwargs = mock_put.call_args
    assert kwargs["headers"] == {"Content-Type": "application/octet-stream"}


def test_transfer_multipart_uploads_each_part_without_headers(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"ab" * 16)  # 32 bytes, 2 parts of 16
    presign = {
        "uploadId": "up_1",
        "uploadUrl": None,
        "partSizeBytes": 16,
        "requiredHeaders": {"Content-Type": "application/octet-stream"},
        "parts": [
            {"partNumber": 1, "url": "https://s3.example/p1"},
            {"partNumber": 2, "url": "https://s3.example/p2"},
        ],
    }
    with patch("nucleus.model_weights.requests.put") as mock_put:
        mock_put.return_value = _ok_put('"etag-x"')
        parts = transfer_weights_to_storage(str(path), presign, 32)

    assert parts == [
        {"partNumber": 1, "eTag": "etag-x"},
        {"partNumber": 2, "eTag": "etag-x"},
    ]
    assert mock_put.call_count == 2
    # Part PUTs are signed without the Content-Type condition — sending
    # requiredHeaders on them makes S3 reject the signature.
    for _, kwargs in mock_put.call_args_list:
        assert kwargs["headers"] == {}


def test_transfer_multipart_raises_without_etag(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"a" * 16)
    presign = {
        "uploadId": "up_1",
        "uploadUrl": None,
        "partSizeBytes": 16,
        "parts": [{"partNumber": 1, "url": "https://s3.example/p1"}],
    }
    with (
        patch("nucleus.model_weights.requests.put") as mock_put,
        pytest.raises(RuntimeError, match="did not return an ETag"),
    ):
        mock_put.return_value = _ok_put(etag="")
        transfer_weights_to_storage(str(path), presign, 16)


def test_transfer_raises_when_presign_has_no_targets(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"a")
    with pytest.raises(ValueError, match="neither an uploadUrl"):
        transfer_weights_to_storage(
            str(path), {"uploadId": "up_1", "uploadUrl": None}, 1
        )


def test_transfer_raises_on_failed_put(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"a")
    response = MagicMock()
    response.ok = False
    response.status_code = 403
    response.text = "AccessDenied"
    with patch("nucleus.model_weights.requests.put") as mock_put:
        mock_put.return_value = response
        with pytest.raises(RuntimeError, match="403"):
            transfer_weights_to_storage(
                str(path),
                {"uploadId": "up_1", "uploadUrl": "https://s3.example/put"},
                1,
            )


def test_transfer_reports_progress(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"a" * 32)
    seen = []
    with patch("nucleus.model_weights.requests.put") as mock_put:
        mock_put.return_value = _ok_put()
        transfer_weights_to_storage(
            str(path),
            {"uploadId": "up_1", "uploadUrl": "https://s3.example/put"},
            32,
            on_progress=lambda sent, total: seen.append((sent, total)),
        )
    assert seen == [(32, 32)]


# --------------------------------------------------------------------------- #
# Download streaming
# --------------------------------------------------------------------------- #
def test_stream_weights_to_file_writes_chunks(tmp_path):
    target = tmp_path / "nested" / "out.bin"
    response = MagicMock()
    response.ok = True
    response.headers = {"Content-Length": "6"}
    response.iter_content.return_value = [b"abc", b"def"]
    response.__enter__ = lambda self: self
    response.__exit__ = lambda *args: False

    with patch("nucleus.model_weights.requests.get", return_value=response):
        written = stream_weights_to_file(
            "https://s3.example/signed-get", str(target)
        )

    assert written == str(target)
    assert target.read_bytes() == b"abcdef"


def test_stream_weights_to_file_raises_on_error(tmp_path):
    response = MagicMock()
    response.ok = False
    response.status_code = 404
    response.__enter__ = lambda self: self
    response.__exit__ = lambda *args: False

    with patch("nucleus.model_weights.requests.get", return_value=response):
        with pytest.raises(RuntimeError, match="404"):
            stream_weights_to_file(
                "https://s3.example/gone", str(tmp_path / "out.bin")
            )


# --------------------------------------------------------------------------- #
# Client methods
# --------------------------------------------------------------------------- #
def test_upload_model_weights_drives_presign_put_finalize(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"x" * 64)
    client = _client()
    client.make_request = MagicMock(
        side_effect=[
            {
                "uploadId": "up_1",
                "uploadUrl": "https://s3.example/put",
                "requiredHeaders": {},
                "parts": None,
                "partSizeBytes": None,
            },
            _WEIGHTS_DTO,
        ]
    )

    with patch("nucleus.model_weights.requests.put") as mock_put:
        mock_put.return_value = _ok_put()
        weights = client.upload_model_weights(_model(client), str(path))

    assert isinstance(weights, ModelWeights)
    assert weights.present is True
    presign_call, finalize_call = client.make_request.call_args_list
    assert presign_call[0][1] == "model/prj_1/weights/presign"
    assert presign_call[0][0] == {
        "declaredSizeBytes": 64,
        "originalFilename": "weights.bin",
    }
    assert finalize_call[0][1] == "model/prj_1/weights/finalize"
    assert finalize_call[0][0] == {"uploadId": "up_1"}


def test_upload_model_weights_accepts_model_id(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"x")
    client = _client()
    client.make_request = MagicMock(
        side_effect=[
            {"uploadId": "up_1", "uploadUrl": "https://s3.example/put"},
            _WEIGHTS_DTO,
        ]
    )
    with patch("nucleus.model_weights.requests.put") as mock_put:
        mock_put.return_value = _ok_put()
        client.upload_model_weights("prj_9", str(path))

    assert (
        client.make_request.call_args_list[0][0][1]
        == "model/prj_9/weights/presign"
    )


def test_upload_model_weights_rejects_oversized_artifact(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"x")
    client = _client()
    client.make_request = MagicMock()

    with (
        patch(
            "nucleus.os.path.getsize", return_value=MODEL_WEIGHTS_MAX_BYTES + 1
        ),
        pytest.raises(ValueError, match="exceeds the 10 GB"),
    ):
        client.upload_model_weights("prj_1", str(path))

    # Rejected before any network call.
    client.make_request.assert_not_called()


def test_download_model_weights_resolves_signed_url(tmp_path):
    target = tmp_path / "out.bin"
    client = _client()
    client.make_request = MagicMock(
        return_value={"url": "https://s3.example/signed-get"}
    )
    with patch(
        "nucleus.stream_weights_to_file", return_value=str(target)
    ) as mock_stream:
        written = client.download_model_weights("prj_1", str(target))

    assert written == str(target)
    assert (
        client.make_request.call_args[0][1]
        == "model/prj_1/weights/download?json=1"
    )
    assert client.make_request.call_args[1]["requests_command"] is requests.get
    mock_stream.assert_called_once_with(
        "https://s3.example/signed-get", str(target), None
    )


def test_download_model_weights_raises_without_url(tmp_path):
    client = _client()
    client.make_request = MagicMock(return_value={})
    with pytest.raises(ValueError, match="no downloadable weights"):
        client.download_model_weights("prj_1", str(tmp_path / "out.bin"))


def test_get_model_weights_parses_dto():
    client = _client()
    client.make_request = MagicMock(return_value=_WEIGHTS_DTO)
    weights = client.get_model_weights("prj_1")

    assert weights.original_filename == "weights.bin"
    assert client.make_request.call_args[0][1] == "model/prj_1/weights"
    assert client.make_request.call_args[1]["requests_command"] is requests.get


def test_delete_model_weights_returns_flag():
    client = _client()
    client.make_request = MagicMock(return_value={"deleted": True})
    assert client.delete_model_weights("prj_1") is True
    assert (
        client.make_request.call_args[1]["requests_command"] is requests.delete
    )


def test_delete_model_weights_false_when_nothing_deleted():
    client = _client()
    client.make_request = MagicMock(return_value={"deleted": False})
    assert client.delete_model_weights("prj_1") is False


# --------------------------------------------------------------------------- #
# Model convenience wrappers
# --------------------------------------------------------------------------- #
def test_model_weights_helpers_delegate_to_client():
    client = MagicMock()
    model = Model("prj_1", "My CNN", "My-CNN", {}, client)

    model.upload_weights("/tmp/w.bin", content_type="application/zip")
    client.upload_model_weights.assert_called_once_with(
        model, "/tmp/w.bin", content_type="application/zip"
    )

    model.download_weights("/tmp/out.bin")
    client.download_model_weights.assert_called_once_with(
        model, "/tmp/out.bin"
    )

    model.weights()
    client.get_model_weights.assert_called_once_with(model)

    model.delete_weights()
    client.delete_model_weights.assert_called_once_with(model)


def test_upload_uses_basename_when_original_filename_unset(tmp_path):
    path = tmp_path / "my-model.safetensors"
    path.write_bytes(b"x")
    client = _client()
    client.make_request = MagicMock(
        side_effect=[
            {"uploadId": "up_1", "uploadUrl": "https://s3.example/put"},
            _WEIGHTS_DTO,
        ]
    )
    with patch("nucleus.model_weights.requests.put") as mock_put:
        mock_put.return_value = _ok_put()
        client.upload_model_weights("prj_1", str(path))

    payload = client.make_request.call_args_list[0][0][0]
    assert payload["originalFilename"] == os.path.basename(str(path))
