"""The weights artifact attached to a model.

:class:`ModelWeights` is the metadata users see; the rest of this module is
internal machinery for :meth:`NucleusClient.upload_model_weights` and
:meth:`NucleusClient.download_model_weights`.
"""

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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

#: Largest weights artifact that can be attached to a model.
MODEL_WEIGHTS_MAX_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB

# Chunks of a large upload sent at once. Each transfer is
# connection-throughput-bound, so a small pool is several times faster.
CONCURRENT_PART_UPLOADS = 4

# Ceiling on how much part data is held in memory at once. Each in-flight part is
# read fully into a `bytes` before it is sent, so peak usage is
# `workers * partSizeBytes` and the part size is chosen by the server — an
# unusually large one would otherwise multiply into GBs of resident memory.
MAX_INFLIGHT_PART_BYTES = 512 * 1024 * 1024  # 512 MB

# Chunk size for streaming a download to disk.
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

# Transfers can run far longer than an API call, so they don't share the API's
# network timeout.
TRANSFER_TIMEOUT_SEC = 60 * 60

# A single PUT/GET can blip on a multi-GB transfer without the whole upload being
# doomed, so transient storage failures (network errors, 429, 5xx) are retried a
# few times with exponential backoff before giving up.
TRANSFER_MAX_ATTEMPTS = 5
TRANSFER_BACKOFF_BASE_SEC = 0.5

#: Called with the cumulative ``bytes_transferred`` as a transfer progresses.
ProgressCallback = Callable[[int], None]


@dataclass
class ModelWeights:
    """Metadata for the weights artifact attached to a model.

    Attributes:
        model_project_id: Id of the model the artifact belongs to (``prj_*``).
        present: Whether the artifact is available to download. ``False`` while
            an upload is still in progress, or if nothing was ever uploaded.
        status: Current state of the artifact, or ``None`` if there isn't one.
        size_bytes: Size of the artifact.
        original_filename: Filename recorded when the artifact was uploaded.
        content_type: Content type recorded when the artifact was uploaded.
        download_url: Temporary URL the artifact can be fetched from. Only set
            when ``present``; prefer
            :meth:`NucleusClient.download_model_weights`, which handles this
            for you.
    """

    model_project_id: Optional[str] = None
    present: bool = False
    status: Optional[str] = None
    size_bytes: Optional[int] = None
    original_filename: Optional[str] = None
    content_type: Optional[str] = None
    download_url: Optional[str] = None

    @classmethod
    def from_json(cls, payload: dict) -> "ModelWeights":
        """Instantiate from an API weights payload."""
        return cls(
            model_project_id=payload.get(MODEL_PROJECT_ID_KEY),
            present=bool(payload.get(PRESENT_KEY, False)),
            status=payload.get(STATUS_KEY),
            size_bytes=payload.get(SIZE_BYTES_KEY),
            original_filename=payload.get(ORIGINAL_FILENAME_KEY),
            content_type=payload.get(CONTENT_TYPE_KEY),
            download_url=payload.get(DOWNLOAD_URL_KEY),
        )


def _strip_quotes(etag: str) -> str:
    return etag.strip('"')


def _read_part(path: str, offset: int, size: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(size)


class _RetryableTransferError(Exception):
    """Marks a storage failure worth retrying (network blip, 429, or 5xx).

    The underlying error is attached as ``__cause__`` so it can be surfaced
    unchanged once the retries are exhausted.
    """


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def _with_retries(
    send: Callable[[], Any], reset: Optional[Callable[[], None]] = None
) -> Any:
    """Run ``send``; retry transient storage failures with exponential backoff.

    ``reset`` runs before each retry to rewind partial state (re-seek the file,
    truncate the temp download) so the retried request sends a full body.
    """
    last_error: _RetryableTransferError = _RetryableTransferError()
    for attempt in range(TRANSFER_MAX_ATTEMPTS):
        try:
            return send()
        except _RetryableTransferError as exc:
            last_error = exc
            if attempt == TRANSFER_MAX_ATTEMPTS - 1:
                break
            if reset is not None:
                reset()
            time.sleep(TRANSFER_BACKOFF_BASE_SEC * 2**attempt)
    raise (
        last_error.__cause__
        if last_error.__cause__ is not None
        else last_error
    )


def _put_bytes(
    url: str,
    body: Any,
    headers: Optional[Dict[str, str]] = None,
    reset: Optional[Callable[[], None]] = None,
) -> str:
    """PUT a body to a presigned S3 URL and return the object/part ETag.

    Transient failures are retried; ``reset`` (for a rewindable ``body``) runs
    before each retry so the resent request isn't truncated.
    """

    def send() -> str:
        try:
            response = requests.put(
                url,
                data=body,
                headers=headers or {},
                timeout=TRANSFER_TIMEOUT_SEC,
            )
        except requests.exceptions.RequestException as exc:
            raise _RetryableTransferError() from exc
        if response.ok:
            return _strip_quotes(response.headers.get("ETag", ""))
        message = (
            f"Upload to storage failed with status {response.status_code}: "
            f"{response.text[:200]}"
        )
        if _is_retryable_status(response.status_code):
            raise _RetryableTransferError() from RuntimeError(message)
        raise RuntimeError(message)

    return _with_retries(send, reset)


def _progress_to_bar(progress_bar: Any) -> ProgressCallback:
    """Adapt the cumulative-bytes callback to a tqdm bar."""

    def update(transferred: int) -> None:
        progress_bar.update(transferred - progress_bar.n)

    return update


class _ProgressReader:
    """File wrapper that reports progress as ``requests`` reads the body.

    Everything except ``read`` is delegated to the wrapped handle, so
    ``requests`` still sizes the body from ``fileno()``/``tell()`` and sends a
    normal Content-Length request rather than a chunked one.
    """

    def __init__(
        self,
        handle: Any,
        total_bytes: int,
        on_progress: ProgressCallback,
    ) -> None:
        self._handle = handle
        self._total_bytes = total_bytes
        self._on_progress = on_progress
        self._transferred = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def reset(self) -> None:
        """Rewind the byte count so a retried PUT re-reports from zero."""
        self._transferred = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        if chunk:
            self._transferred += len(chunk)
            self._on_progress(min(self._transferred, self._total_bytes))
        return chunk


def _upload_single(
    path: str,
    upload_url: str,
    headers: Dict[str, str],
    total_bytes: int,
    on_progress: Optional[ProgressCallback],
) -> None:
    with open(path, "rb") as handle:
        # A single PUT is one request no matter its size, so progress has to come
        # from the read side — otherwise a caller's progress bar sits at 0% for
        # the whole transfer and then jumps to 100%.
        body = (
            handle
            if on_progress is None
            else _ProgressReader(handle, total_bytes, on_progress)
        )

        def reset() -> None:
            # A retry re-sends from the top, so rewind the handle (and the
            # progress count) or the resent body would be truncated.
            handle.seek(0)
            if isinstance(body, _ProgressReader):
                body.reset()

        _put_bytes(upload_url, body, headers, reset=reset)
    if on_progress is not None:
        on_progress(total_bytes)


def _part_upload_workers(part_count: int, part_size_bytes: int) -> int:
    """Concurrency for a multipart upload, bounded by memory as well as count.

    Always at least 1: a part larger than the whole budget still has to be sent,
    and one at a time is the least memory that can do it.
    """
    budget = max(1, MAX_INFLIGHT_PART_BYTES // max(1, part_size_bytes))
    return max(1, min(CONCURRENT_PART_UPLOADS, part_count, budget))


def _upload_multipart(
    path: str,
    parts: List[dict],
    part_size_bytes: int,
    total_bytes: int,
    on_progress: Optional[ProgressCallback],
) -> List[Dict[str, Any]]:
    """Upload each part concurrently and return the finalize part list."""
    transferred = 0
    # Parts upload concurrently, and `transferred += ...` is a read-modify-write
    # the interpreter can interleave. The callback is invoked under the same lock
    # as the counter, not just the arithmetic: releasing first lets two threads
    # compute 100 and 200 and then call in either order, so a caller's progress
    # bar can jump backwards.
    progress_lock = threading.Lock()

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
        with progress_lock:
            transferred += len(chunk)
            if on_progress is not None:
                on_progress(min(transferred, total_bytes))
        return {PART_NUMBER_KEY: part_number, ETAG_KEY: etag}

    with ThreadPoolExecutor(
        max_workers=_part_upload_workers(len(parts), part_size_bytes)
    ) as pool:
        finalized = list(pool.map(upload_part, parts))

    return sorted(finalized, key=lambda p: p[PART_NUMBER_KEY])


def _transfer_weights_to_storage(
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


def _download_to_handle(
    url: str, handle: Any, on_progress: Optional[ProgressCallback]
) -> None:
    """GET a signed URL and stream the body into an open file handle.

    A network blip, 5xx, or short read raises :class:`_RetryableTransferError`
    so the caller can retry from a truncated handle.
    """
    with requests.get(
        url, stream=True, timeout=TRANSFER_TIMEOUT_SEC
    ) as response:
        if not response.ok:
            message = (
                f"Download from storage failed with status "
                f"{response.status_code}"
            )
            if _is_retryable_status(response.status_code):
                raise _RetryableTransferError() from RuntimeError(message)
            raise RuntimeError(message)
        total_bytes = int(response.headers.get("Content-Length") or 0)
        transferred = 0
        try:
            for chunk in response.iter_content(
                chunk_size=DOWNLOAD_CHUNK_BYTES
            ):
                if not chunk:
                    continue
                handle.write(chunk)
                transferred += len(chunk)
                if on_progress is not None:
                    on_progress(transferred)
        except requests.exceptions.RequestException as exc:
            raise _RetryableTransferError() from exc
        # A stream can end short of Content-Length without raising (e.g. a
        # dropped connection); retry rather than promote a truncated artifact.
        # Only checkable when the length was advertised.
        if total_bytes and transferred != total_bytes:
            raise _RetryableTransferError() from RuntimeError(
                f"Download incomplete: received {transferred} of "
                f"{total_bytes} bytes"
            )


def _stream_weights_to_file(
    url: str,
    path: str,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """Stream a signed download URL to ``path``, returning the path written."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Stream into a sibling temp file and rename only once the body is fully
    # written. Writing straight to `path` would leave a truncated artifact
    # behind on an interrupted transfer — indistinguishable from a complete one
    # until something tries to load the weights.
    handle_fd, partial_path = tempfile.mkstemp(
        dir=parent or None,
        prefix=f"{os.path.basename(path)}.",
        suffix=".part",
    )
    try:
        with os.fdopen(handle_fd, "wb") as handle:

            def send() -> None:
                _download_to_handle(url, handle, on_progress)

            def reset() -> None:
                # Each attempt re-streams the whole body, so discard whatever a
                # failed attempt wrote before retrying.
                handle.seek(0)
                handle.truncate()

            _with_retries(send, reset)
        # Same directory, so this is an atomic replace.
        os.replace(partial_path, path)
    except BaseException:
        try:
            os.remove(partial_path)
        except OSError:
            pass
        raise
    return path


def _presign_payload(
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


def _finalize_payload(
    upload_id: str, parts: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Build the finalize request body."""
    payload: Dict[str, Any] = {UPLOAD_ID_KEY: upload_id}
    if parts:
        payload[PARTS_KEY] = parts
    return payload
