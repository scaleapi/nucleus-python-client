import asyncio
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    Callable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import aiohttp
import nest_asyncio
from tqdm import tqdm

from nucleus.constants import DEFAULT_NETWORK_TIMEOUT_SEC
from nucleus.errors import NucleusAPIError
from nucleus.retry_strategy import RetryStrategy

from .logger import logger

if TYPE_CHECKING:
    from . import NucleusClient


@dataclass
class FileFormField:
    name: str
    filename: str
    value: BinaryIO
    content_type: str


FileFormData = Sequence[FileFormField]

UPLOAD_SEMAPHORE = asyncio.Semaphore(10)


class FormDataContextHandler:
    """A context handler for file form data that handles closing all files in a request.

    Why do I need to wrap my requests in such a funny way?

    1. Form data must be regenerated on each request to avoid errors
        see https://github.com/Rapptz/discord.py/issues/6531
    2. Files must be properly open/closed for each request.
    3. We need to be able to do 1/2 above multiple times so that we can implement retries
        properly.

    Write a function that returns a tuple of form data and file pointers, then pass it to the
    constructor of this class, and this class will handle the rest for you.
    """

    def __init__(
        self,
        form_data_and_file_pointers_fn: Callable[
            ..., Tuple[FileFormData, Sequence[BinaryIO]]
        ],
    ):
        self._form_data_and_file_pointer_fn = form_data_and_file_pointers_fn
        self._file_pointers = None

    def __enter__(self):
        (
            file_form_data,
            self._file_pointers,
        ) = self._form_data_and_file_pointer_fn()
        form = aiohttp.FormData()
        for field in file_form_data:
            form.add_field(
                name=field.name,
                filename=field.filename,
                value=field.value,
                content_type=field.content_type,
            )
        return form

    def __exit__(self, exc_type, exc_val, exc_tb):
        for file_pointer in self._file_pointers:
            file_pointer.close()


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # no event loop running:
        loop = asyncio.new_event_loop()
    else:
        nest_asyncio.apply(loop)
    return loop


def make_multiple_requests_concurrently(
    client: "NucleusClient",
    requests: Sequence[Union[FormDataContextHandler, str]],
    route: Optional[str],
    progressbar: tqdm,
):
    """
    Makes an async post request with form data to a Nucleus endpoint.

    Args:
        client: The client to use for the request.
        requests: a list of requests to make. This list either comprises a string of endpoints to request,
        or a list of FormDataContextHandler object which will handle generating form data, and opening/closing files for each request.
        route: A route is required when requests are for Form Data Post requests
        progressbar: A tqdm progress bar to use for showing progress to the user.
    """
    loop = get_event_loop()
    return loop.run_until_complete(
        _request_helper(client, requests, route, progressbar)
    )


async def _request_helper(
    client: "NucleusClient",
    requests: Sequence[Union[FormDataContextHandler, str]],
    route: Optional[str],
    progressbar: tqdm,
):
    """
    Makes an async requests to a Nucleus endpoint.

    Args:
        client: The client to use for the request.
        requests: a list of requests to make. This list either comprises a string of endpoints to request,
        or a list of FormDataContextHandler object which will handle generating form data, and opening/closing files for each request.
        route: route for the request.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for request in requests:
            if isinstance(request, FormDataContextHandler):
                assert (
                    route
                ), "A route must be specified for FormDataContextHandler requests"
                req = asyncio.ensure_future(
                    _post_form_data(
                        client=client,
                        request=request,
                        route=route,
                        session=session,
                        progressbar=progressbar,
                    )
                )
                tasks.append(req)
            else:
                req = asyncio.ensure_future(
                    _make_request(
                        client=client,
                        request=request,
                        session=session,
                        progressbar=progressbar,
                    )
                )
                tasks.append(req)

        return await asyncio.gather(*tasks)


async def _post_form_data(
    client: "NucleusClient",
    request: FormDataContextHandler,
    route: str,
    session: aiohttp.ClientSession,
    progressbar: tqdm,
):
    """
    Makes an async post request with files to a Nucleus endpoint.

    Args:
        client: The client to use for the request.
        request: The request to make (See FormDataContextHandler for more details.)
        route: route for the request.
        session: The session to use for the request.
    """
    endpoint = f"{client.endpoint}/{route}"
    logger.info("Posting to %s", endpoint)

    async with UPLOAD_SEMAPHORE:
        for sleep_time in RetryStrategy.sleep_times() + [-1]:
            with request as form:
                async with session.post(
                    endpoint,
                    data=form,
                    auth=aiohttp.BasicAuth(client.api_key, ""),
                    timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
                ) as response:
                    data = await _parse_async_response(
                        endpoint, session, response, sleep_time
                    )
                    if data is None:
                        continue

                    progressbar.update(1)
                    return data


async def _make_request(
    client: "NucleusClient",
    request: str,
    session: aiohttp.ClientSession,
    progressbar: tqdm,
):
    """
    Makes an async post request with files to a Nucleus endpoint.

    Args:
        client: The client to use for the request.
        request: The request to make (See FormDataContextHandler for more details.)
        route: route for the request.
        session: The session to use for the request.

    Returns:
        A tuple (endpoint request string, response from endpoint)
    """
    endpoint = f"{client.endpoint}/{request}"
    logger.info("GET %s", endpoint)

    async with UPLOAD_SEMAPHORE:
        for sleep_time in RetryStrategy.sleep_times() + [-1]:
            async with session.get(
                endpoint,
                auth=aiohttp.BasicAuth(client.api_key, ""),
                timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
            ) as response:
                data = await _parse_async_response(
                    endpoint, session, response, sleep_time
                )
                if data is None:
                    continue

                progressbar.update(1)
                return (request, data)


async def _parse_async_response(endpoint, session, response, sleep_time):
    logger.info("API request has response code %s", response.status)

    try:
        data = await response.json()
    except aiohttp.client_exceptions.ContentTypeError:
        # In case of 404, the server returns text
        data = await response.text()
    if response.status in RetryStrategy.statuses and sleep_time != -1:
        time.sleep(sleep_time)
        return None

    if response.status == 503:
        raise TimeoutError(
            "The request to upload your max is timing out, please lower local_files_per_upload_request in your api call."
        )

    if not response.ok:
        raise NucleusAPIError(
            endpoint,
            session.get,
            aiohttp_response=(
                response.status,
                response.reason,
                data,
            ),
        )

    return data
