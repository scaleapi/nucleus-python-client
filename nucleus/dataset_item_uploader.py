import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    List,
    Sequence,
    Tuple,
)

from nucleus.async_utils import (
    FileFormData,
    FileFormField,
    FormDataContextHandler,
    make_many_form_data_requests_concurrently,
)

from .constants import DATASET_ID_KEY, IMAGE_KEY, ITEMS_KEY, UPDATE_KEY
from .dataset_item import DatasetItem
from .errors import NotFoundError
from .payload_constructor import construct_append_payload
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
        batch_size: int = 5000,
        update: bool = False,
        local_files_per_upload_request: int = 10,
        local_file_upload_concurrency: int = 30,
    ) -> UploadResponse:
        """

        Args:
            dataset_items: Items to Upload
            batch_size: How many items to pool together for a single request for items
             without files to upload
            files_per_upload_request: How many items to pool together for a single
                request for items with files to upload

            update: Update records instead of overwriting

        Returns:

        """
        local_items = []
        remote_items = []
        if local_files_per_upload_request > 10:
            raise ValueError("local_files_per_upload_request should be <= 10")

        # Check local files exist before sending requests
        for item in dataset_items:
            if item.local:
                if not item.local_file_exists():
                    raise NotFoundError()
                local_items.append(item)
            else:
                remote_items.append(item)

        agg_response = UploadResponse(json={DATASET_ID_KEY: self.dataset_id})

        async_responses: List[Any] = []

        if local_items:
            async_responses.extend(
                self._process_append_requests_local(
                    self.dataset_id,
                    items=local_items,
                    update=update,
                    batch_size=batch_size,
                    local_files_per_upload_request=local_files_per_upload_request,
                    local_file_upload_concurrency=local_file_upload_concurrency,
                )
            )

        remote_batches = [
            remote_items[i : i + batch_size]
            for i in range(0, len(remote_items), batch_size)
        ]

        if remote_batches:
            tqdm_remote_batches = self._client.tqdm_bar(
                remote_batches, desc="Remote file batches"
            )
            for batch in tqdm_remote_batches:
                responses = self._process_append_requests(
                    dataset_id=self.dataset_id,
                    payload=construct_append_payload(batch, update),
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
        items: Sequence[DatasetItem],
        update: bool,
        batch_size: int,
        local_files_per_upload_request: int,
        local_file_upload_concurrency: int,
    ):
        # Batch into requests
        requests = []
        batch_size = local_files_per_upload_request
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            request = FormDataContextHandler(
                self.get_form_data_and_file_pointers_fn(batch, update)
            )
            requests.append(request)

        progressbar = self._client.tqdm_bar(
            total=len(requests), desc="Local file batches"
        )

        return make_many_form_data_requests_concurrently(
            self._client,
            requests,
            f"dataset/{dataset_id}/append",
            progressbar=progressbar,
            concurrency=local_file_upload_concurrency,
        )

    def _process_append_requests(
        self,
        dataset_id: str,
        payload: dict,
        update: bool,
        batch_size: int = 20,
    ):
        items = payload[ITEMS_KEY]
        payloads = [
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

    def get_form_data_and_file_pointers_fn(
        self, items: Sequence[DatasetItem], update: bool
    ) -> Callable[..., Tuple[FileFormData, Sequence[BinaryIO]]]:
        """Defines a function to be called on each retry.

        File pointers are also returned so whoever calls this function can
        appropriately close the files. This is intended for use with a
        FormDataContextHandler in order to make form data requests.
        """

        def fn():

            # For some reason, our backend only accepts this reformatting of items when
            # doing local upload.
            # TODO: make it just accept the same exact format as a normal append request
            # i.e. the output of construct_append_payload(items, update)
            json_data = []
            for item in items:
                item_payload = item.to_payload()
                item_payload[UPDATE_KEY] = update
                json_data.append(item_payload)

            form_data = [
                FileFormField(
                    name=ITEMS_KEY,
                    filename=None,
                    value=json.dumps(json_data, allow_nan=False),
                    content_type="application/json",
                )
            ]

            file_pointers = []
            for item in items:
                # I don't know of a way to use with, since all files in the request
                # need to be opened at the same time.
                # pylint: disable=consider-using-with
                image_fp = open(item.image_location, "rb")
                # pylint: enable=consider-using-with
                img_type = f"image/{os.path.splitext(item.image_location)[1].strip('.')}"
                form_data.append(
                    FileFormField(
                        name=IMAGE_KEY,
                        filename=item.image_location,
                        value=image_fp,
                        content_type=img_type,
                    )
                )
                file_pointers.append(image_fp)
            return form_data, file_pointers

        return fn
