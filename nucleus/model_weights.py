"""Model weights artifacts: metadata plus the direct-to-S3 transfer helpers.

Bytes never transit the Nucleus API. Upload is a three-step flow — presign
against the API, ``PUT`` straight to the returned S3 URL(s), then finalize —
and download resolves a short-lived signed URL and streams from it. The
``NucleusClient`` methods (:meth:`NucleusClient.upload_model_weights`,
:meth:`NucleusClient.download_model_weights`) drive the helpers here.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import requests

from .constants import (
    CHECKSUM_SHA256_KEY,
    CONTENT_TYPE_KEY,
    DECLARED_SIZE_BYTES_KEY,
    DOWNLOAD_URL_KEY,
    ETAG_KEY,
    MODEL_PROJECT_ID_KEY,
    ORIGINAL_FILENAME_KEY,
    PART_NUMBER_KEY,
    PART_SIZE_BYTES_KEY,
    PARTS_KEY,
    PRESENT_KEY,
    REQUIRED_HEADERS_KEY,
    SIZE_BYTES_KEY,
    STATUS_KEY,
    UPLOAD_ID_KEY,
    UPLOAD_URL_KEY,
    URL_KEY,
)

if TYPE_CHECKING:
    from . import NucleusClient

#: Hard cap the server enforces on a single artifact; checked client-side so a
#: multi-GB read isn't started just to be rejected by presign.
MODEL_WEIGHTS_MAX_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB

#: Parts in flight at once. A single S3 PUT is connection-throughput-bound, so a
#: small pool is several times faster on multi-GB artifacts.
CONCURRENT_PART_UPLOADS = 4

#: Chunk size for streaming a download to disk.
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

#: Upload/download PUTs and GETs go straight to S3 and can take far longer than
#: an API call, so they don't use the API's network timeout.
TRANSFER_TIMEOUT_SEC = 60 * 60

#: Called with ``(bytes_transferred, total_bytes)`` as a transfer progresses.
ProgressCallback = Callable[[int, int], None]


@dataclass
class ModelWeights:
    """Metadata for the weights artifact attached to a model.

    Attributes:
        model_project_id: Id of the model the artifact belongs to (``prj_*``).
        present: Whether a *ready* artifact exists. ``False`` while an upload is
            still pending or if none was ever uploaded.
        status: Raw server-side status, or ``None`` when no artifact exists.
        size_bytes: Size of the stored artifact.
        original_filename: Filename supplied at upload time.
        content_type: Content type supplied at upload time.
        download_url: Short-lived signed URL, populated only when ``present``.
    """

    model_project_id: Optional[str] = None
    present: bool = False
    status: Optional[str] = None
    size_bytes: Optional[int] = None
    original_filename: Optional[str] = None
    content_type: Optional[str] = None
    download_url: Optional[str] = None
    _client: Optional["NucleusClient"] = field(repr=False, default=None)

    @classmethod
    def from_json(
        cls, payload: dict, client: Optional["NucleusClient"] = None
    ) -> "ModelWeights":
        """Instantiate from the server's weights DTO."""
        return cls(
            model_project_id=payload.get(MODEL_PROJECT_ID_KEY),
            present=bool(payload.get(PRESENT_KEY, False)),
            status=payload.get(STATUS_KEY),
            size_bytes=payload.get(SIZE_BYTES_KEY),
            original_filename=payload.get(ORIGINAL_FILENAME_KEY),
            content_type=payload.get(CONTENT_TYPE_KEY),
            download_url=payload.get(DOWNLOAD_URL_KEY),
            _client=client,
        )


def _strip_quotes(etag: str) -> str:
    return etag.strip('"')


def _read_part(path: str, offset: int, size: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(size)


def _put_bytes(
    url: str, body: Any, headers: Optional[Dict[str, str]] = None
) -> str:
    """PUT a body to a presigned S3 URL and return the object/part ETag."""
    response = requests.put(
        url,
        data=body,
        headers=headers or {},
        timeout=TRANSFER_TIMEOUT_SEC,
    )
    if not response.ok:
        raise RuntimeError(
            f"Upload to storage failed with status {response.status_code}: "
            f"{response.text[:200]}"
        )
    return _strip_quotes(response.headers.get("ETag", ""))


def _upload_single(
    path: str,
    upload_url: str,
    headers: Dict[str, str],
    total_bytes: int,
    on_progress: Optional[ProgressCallback],
) -> None:
    with open(path, "rb") as handle:
        _put_bytes(upload_url, handle, headers)
    if on_progress is not None:
        on_progress(total_bytes, total_bytes)


def _upload_multipart(
    path: str,
    parts: List[dict],
    part_size_bytes: int,
    total_bytes: int,
    on_progress: Optional[ProgressCallback],
) -> List[Dict[str, Any]]:
    """Upload each part concurrently and return the finalize part list."""
    transferred = 0
    finalized: List[Dict[str, Any]] = []

    def upload_part(part: dict) -> Dict[str, Any]:
        nonlocal transferred
        part_number = int(part[PART_NUMBER_KEY])
        offset = (part_number - 1) * part_size_bytes
        chunk = _read_part(path, offset, part_size_bytes)
        # Part PUTs are signed without the Content-Type condition, so they must
        # be sent with no extra headers — including none of `requiredHeaders`.
        etag = _put_bytes(part[URL_KEY], chunk)
        if not etag:
            raise RuntimeError(
                f"Storage did not return an ETag for part {part_number}; "
                "cannot finalize the multipart upload"
            )
        transferred += len(chunk)
        if on_progress is not None:
            on_progress(min(transferred, total_bytes), total_bytes)
        return {PART_NUMBER_KEY: part_number, ETAG_KEY: etag}

    with ThreadPoolExecutor(
        max_workers=min(CONCURRENT_PART_UPLOADS, len(parts))
    ) as pool:
        finalized = list(pool.map(upload_part, parts))

    return sorted(finalized, key=lambda p: p[PART_NUMBER_KEY])


def transfer_weights_to_storage(
    path: str,
    presign: dict,
    total_bytes: int,
    on_progress: Optional[ProgressCallback] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Send the file to storage using a presign response.

    Returns the part list to hand to finalize for a multipart upload, or
    ``None`` when the artifact went up as a single PUT.
    """
    upload_url = presign.get(UPLOAD_URL_KEY)
    if upload_url:
        _upload_single(
            path,
            upload_url,
            presign.get(REQUIRED_HEADERS_KEY) or {},
            total_bytes,
            on_progress,
        )
        return None

    parts = presign.get(PARTS_KEY)
    part_size_bytes = presign.get(PART_SIZE_BYTES_KEY)
    if not parts or not part_size_bytes:
        raise ValueError(
            "Presign response contained neither an uploadUrl nor multipart "
            "parts; cannot upload"
        )
    return _upload_multipart(
        path, list(parts), int(part_size_bytes), total_bytes, on_progress
    )


def stream_weights_to_file(
    url: str,
    path: str,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """Stream a signed download URL to ``path``, returning the path written."""
    with requests.get(
        url, stream=True, timeout=TRANSFER_TIMEOUT_SEC
    ) as response:
        if not response.ok:
            raise RuntimeError(
                f"Download from storage failed with status "
                f"{response.status_code}"
            )
        total_bytes = int(response.headers.get("Content-Length") or 0)
        transferred = 0
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as handle:
            for chunk in response.iter_content(
                chunk_size=DOWNLOAD_CHUNK_BYTES
            ):
                if not chunk:
                    continue
                handle.write(chunk)
                transferred += len(chunk)
                if on_progress is not None:
                    on_progress(transferred, total_bytes)
    return path


def presign_payload(
    declared_size_bytes: int,
    content_type: Optional[str],
    original_filename: Optional[str],
    checksum_sha256: Optional[str],
) -> Dict[str, Any]:
    """Build the presign request body, omitting unset optional fields."""
    payload: Dict[str, Any] = {
        DECLARED_SIZE_BYTES_KEY: declared_size_bytes,
    }
    if content_type is not None:
        payload[CONTENT_TYPE_KEY] = content_type
    if original_filename is not None:
        payload[ORIGINAL_FILENAME_KEY] = original_filename
    if checksum_sha256 is not None:
        payload[CHECKSUM_SHA256_KEY] = checksum_sha256
    return payload


def finalize_payload(
    upload_id: str, parts: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Build the finalize request body."""
    payload: Dict[str, Any] = {UPLOAD_ID_KEY: upload_id}
    if parts:
        payload[PARTS_KEY] = parts
    return payload
