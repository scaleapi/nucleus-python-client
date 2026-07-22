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
    make_multiple_requests_concurrently,
)

from .constants import IMAGE_KEY, ITEMS_KEY, UPDATE_KEY
from .dataset_item import DatasetItem
from .errors import NotFoundError

if TYPE_CHECKING:
    from . import NucleusClient


class DatasetItemUploader:
    def __init__(self, dataset_id: str, client: "NucleusClient"):  # noqa: F821
        self.dataset_id = dataset_id
        self._client = client

    def upload_local_async(
        self,
        dataset_items: List[DatasetItem],
        update: bool = False,
        local_files_per_upload_request: int = 10,
    ) -> List[Any]:
        """Uploads local files as multipart to the async append endpoint.

        Returns a list of job responses (one per batch).
        """
        if local_files_per_upload_request > 10:
            raise ValueError("local_files_per_upload_request should be <= 10")

        for item in dataset_items:
            if item.local and not item.local_file_exists():
                raise NotFoundError()

        requests = []
        batch_size = local_files_per_upload_request
        for i in range(0, len(dataset_items), batch_size):
            batch = dataset_items[i : i + batch_size]
            request = FormDataContextHandler(
                self._build_form_data_fn(batch, update)
            )
            requests.append(request)

        progressbar = self._client.tqdm_bar(
            total=len(requests),
            desc=f"Uploading {len(dataset_items)} items in {len(requests)} batches",
        )

        return make_multiple_requests_concurrently(
            self._client,
            requests,
            f"dataset/{self.dataset_id}/append?async=1",
            progressbar=progressbar,
        )

    def _build_form_data_fn(
        self, items: Sequence[DatasetItem], update: bool
    ) -> Callable[..., Tuple[FileFormData, Sequence[BinaryIO]]]:
        """Returns a function that builds form data and opens file pointers.

        Called on each retry attempt by FormDataContextHandler to ensure
        file pointers are fresh.
        """

        def fn():
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
