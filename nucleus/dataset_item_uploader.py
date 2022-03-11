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
                responses = self._process_append_requests_local(
                    self.dataset_id, items=batch, update=update
                )
                async_responses.extend(responses)

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
        local_batch_size: int = 10,
    ):
        requests = []
        for i in range(0, len(items), local_batch_size):
            batch = items[i : i + local_batch_size]
            request = FormDataContextHandler(
                self.get_form_data_and_file_pointers_fn(batch, update)
            )
            requests.append(request)

        return make_many_form_data_requests_concurrently(
            self._client,
            requests,
            f"dataset/{dataset_id}/append",
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
        """Constructs a function that will generate form data on each retry."""

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
                image_fp = open(  # pylint: disable=R1732
                    item.image_location, "rb"  # pylint: disable=R1732
                )  # pylint: disable=R1732
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
