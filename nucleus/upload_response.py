from typing import Set
from .dataset_item import DatasetItem
from .constants import (
    NEW_ITEMS,
    UPDATED_ITEMS,
    IGNORED_ITEMS,
    ERROR_ITEMS,
    ERROR_CODES,
    ERROR_PAYLOAD,
    DATASET_ID_KEY,
)


def json_list_to_dataset_item(item_list):
    return [DatasetItem.from_json(item) for item in item_list]


class UploadResponse:
    """
    Response for long upload job
    # TODO refactor
    """

    def __init__(self, json: dict):
        dataset_id = json.get(DATASET_ID_KEY)
        new_items = json.get(NEW_ITEMS, 0)
        updated_items = json.get(UPDATED_ITEMS, 0)
        ignored_items = json.get(IGNORED_ITEMS, 0)
        upload_errors = json.get(ERROR_ITEMS, 0)
        upload_error_payload = json_list_to_dataset_item(
            json.get(ERROR_PAYLOAD, [])
        )

        self.dataset_id = dataset_id
        self.new_items = new_items
        self.updated_items = updated_items
        self.ignored_items = ignored_items
        self.upload_errors = upload_errors
        self.error_codes: Set[str] = set()
        self.error_payload = upload_error_payload

    def update_response(self, json):
        """
        :param json: { new_items: int, updated_items: int, ignored_items: int, upload_errors: int, }
        """
        assert self.dataset_id == json.get(DATASET_ID_KEY)
        self.new_items += json.get(NEW_ITEMS, 0)
        self.updated_items += json.get(UPDATED_ITEMS, 0)
        self.ignored_items += json.get(IGNORED_ITEMS, 0)
        self.upload_errors += json.get(ERROR_ITEMS, 0)
        if json.get(ERROR_PAYLOAD, None):
            self.error_payload.extend(json.get(ERROR_PAYLOAD, None))

    def record_error(self, response, num_uploads):
        """
        :param response: json response
        :param num_uploads: len of itemss tried to upload
        """
        status = response.status_code
        self.error_codes.add(status)
        self.upload_errors += num_uploads

    def json(self) -> dict:
        """
        return: { new_items: int, updated_items: int, ignored_items: int, upload_errors: int, }
        """
        result = {
            DATASET_ID_KEY: self.dataset_id,
            NEW_ITEMS: self.new_items,
            UPDATED_ITEMS: self.updated_items,
            IGNORED_ITEMS: self.ignored_items,
            ERROR_ITEMS: self.upload_errors,
        }
        if self.error_payload:
            result[ERROR_PAYLOAD] = self.error_payload

        if self.error_codes:
            result[ERROR_ITEMS] = self.upload_errors
            result[ERROR_CODES] = self.error_codes
        return result
