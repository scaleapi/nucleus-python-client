import asyncio
import json
import os
import time
from typing import TYPE_CHECKING, Any, List

import aiohttp
import nest_asyncio

from .constants import (
    DATASET_ID_KEY,
    DEFAULT_NETWORK_TIMEOUT_SEC,
    IMAGE_KEY,
    IMAGE_URL_KEY,
    ITEMS_KEY,
    UPDATE_KEY,
)
from .dataset_item import DatasetItem
from .errors import NotFoundError
from .logger import logger
from .payload_constructor import construct_append_payload
from .retry_strategy import RetryStrategy
from .upload_response import UploadResponse

if TYPE_CHECKING:
    from . import NucleusClient


class DatasetItemUploader:
    def __init__(self, dataset_id: str, client: "NucleusClient"):  # noqa: F821
        self.dataset_id = dataset_id
        self._client = client

    def upload(
        self,
        dataset_items: List[DatasetItem],
        batch_size: int = 20,
        update: bool = False,
    ) -> UploadResponse:
        """

        Args:
            dataset_items: Items to Upload
            batch_size: How many items to pool together for a single request
            update: Update records instead of overwriting

        Returns:

        """
        local_items = []
        remote_items = []

        # Check local files exist before sending requests
        for item in dataset_items:
            if item.local:
                if not item.local_file_exists():
                    raise NotFoundError()
                local_items.append(item)
            else:
                remote_items.append(item)

        local_batches = [
            local_items[i : i + batch_size]
            for i in range(0, len(local_items), batch_size)
        ]

        remote_batches = [
            remote_items[i : i + batch_size]
            for i in range(0, len(remote_items), batch_size)
        ]

        agg_response = UploadResponse(json={DATASET_ID_KEY: self.dataset_id})

        async_responses: List[Any] = []

        if local_batches:
            tqdm_local_batches = self._client.tqdm_bar(
                local_batches, desc="Local file batches"
            )

            for batch in tqdm_local_batches:
                payload = construct_append_payload(batch, update)
                responses = self._process_append_requests_local(
                    self.dataset_id, payload, update
                )
                async_responses.extend(responses)

        if remote_batches:
            tqdm_remote_batches = self._client.tqdm_bar(
                remote_batches, desc="Remote file batches"
            )
            for batch in tqdm_remote_batches:
                payload = construct_append_payload(batch, update)
                responses = self._process_append_requests(
                    dataset_id=self.dataset_id,
                    payload=payload,
                    update=update,
                    batch_size=batch_size,
                )
                async_responses.extend(responses)

        for response in async_responses:
            agg_response.update_response(response)

        return agg_response

    def _process_append_requests_local(
        self,
        dataset_id: str,
        payload: dict,
        update: bool,  # TODO: understand how to pass this in.
        local_batch_size: int = 10,
    ):
        def get_files(batch):
            for item in batch:
                item[UPDATE_KEY] = update
            request_payload = [
                (
                    ITEMS_KEY,
                    (
                        None,
                        json.dumps(batch, allow_nan=False),
                        "application/json",
                    ),
                )
            ]
            for item in batch:
                image = open(  # pylint: disable=R1732
                    item.get(IMAGE_URL_KEY), "rb"  # pylint: disable=R1732
                )  # pylint: disable=R1732
                img_name = os.path.basename(image.name)
                img_type = (
                    f"image/{os.path.splitext(image.name)[1].strip('.')}"
                )
                request_payload.append(
                    (IMAGE_KEY, (img_name, image, img_type))
                )
            return request_payload

        items = payload[ITEMS_KEY]
        responses: List[Any] = []
        files_per_request = []
        payload_items = []
        for i in range(0, len(items), local_batch_size):
            batch = items[i : i + local_batch_size]
            files_per_request.append(get_files(batch))
            payload_items.append(batch)

        future = self.make_many_files_requests_asynchronously(
            files_per_request,
            f"dataset/{dataset_id}/append",
        )

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:  # no event loop running:
            loop = asyncio.new_event_loop()
            responses = loop.run_until_complete(future)
        else:
            nest_asyncio.apply(loop)
            return loop.run_until_complete(future)

        def close_files(request_items):
            for item in request_items:
                # file buffer in location [1][1]
                if item[0] == IMAGE_KEY:
                    item[1][1].close()

        # don't forget to close all open files
        for p in files_per_request:
            close_files(p)

        return responses

    async def make_many_files_requests_asynchronously(
        self, files_per_request, route
    ):
        """
        Makes an async post request with files to a Nucleus endpoint.

        :param files_per_request: A list of lists of tuples (name, (filename, file_pointer, content_type))
           name will become the name by which the multer can build an array.
        :param route: route for the request
        :return: awaitable list(response)
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.ensure_future(
                    self._make_files_request(
                        files=files, route=route, session=session
                    )
                )
                for files in files_per_request
            ]
            return await asyncio.gather(*tasks)

    async def _make_files_request(
        self,
        files,
        route: str,
        session: aiohttp.ClientSession,
        retry_attempt=0,
        max_retries=3,
        sleep_intervals=(1, 3, 9),
    ):
        """
        Makes an async post request with files to a Nucleus endpoint.

        :param files: A list of tuples (name, (filename, file_pointer, file_type))
        :param route: route for the request
        :param session: Session to use for post.
        :return: response
        """
        endpoint = f"{self._client.endpoint}/{route}"

        logger.info("Posting to %s", endpoint)

        form = aiohttp.FormData()

        for file in files:
            form.add_field(
                name=file[0],
                filename=file[1][0],
                value=file[1][1],
                content_type=file[1][2],
            )

        for sleep_time in RetryStrategy.sleep_times + [-1]:

            async with session.post(
                endpoint,
                data=form,
                auth=aiohttp.BasicAuth(self._client.api_key, ""),
                timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
            ) as response:
                logger.info(
                    "API request has response code %s", response.status
                )

                try:
                    data = await response.json()
                except aiohttp.client_exceptions.ContentTypeError:
                    # In case of 404, the server returns text
                    data = await response.text()
                if (
                    response.status in RetryStrategy.statuses
                    and sleep_time != -1
                ):
                    time.sleep(sleep_time)
                    continue

                if not response.ok:
                    if retry_attempt < max_retries:
                        time.sleep(sleep_intervals[retry_attempt])
                        retry_attempt += 1
                        return self._make_files_request(
                            files,
                            route,
                            session,
                            retry_attempt,
                            max_retries,
                            sleep_intervals,
                        )
                    else:
                        self._client.handle_bad_response(
                            endpoint,
                            session.post,
                            aiohttp_response=(
                                response.status,
                                response.reason,
                                data,
                            ),
                        )

                return data

    def _process_append_requests(
        self,
        dataset_id: str,
        payload: dict,
        update: bool,
        batch_size: int = 20,
    ):
        items = payload[ITEMS_KEY]
        payloads = [
            # batch_size images per request
            {ITEMS_KEY: items[i : i + batch_size], UPDATE_KEY: update}
            for i in range(0, len(items), batch_size)
        ]

        return [
            self._client.make_request(
                payload,
                f"dataset/{dataset_id}/append",
            )
            for payload in payloads
        ]
