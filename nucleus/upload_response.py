NEW_ITEMS = "new_items"
UPDATED_ITEMS = "updated_items"
IGNORED_ITEMS = "ignored_items"
ERROR_ITEMS = "upload_errors"
ERROR_CODES = "error_codes"

class UploadResponse:

    def __init__(self, json={}):
        dataset_id = json.get("dataset_id")
        new_items = json.get(NEW_ITEMS)
        updated_items = json.get(UPDATED_ITEMS)
        ignored_items = json.get(IGNORED_ITEMS)
        upload_errors = json.get(ERROR_ITEMS)

        self.dataset_id = dataset_id
        self.new_items = new_items if new_items else 0
        self.updated_items = updated_items if updated_items else 0
        self.ignored_items = ignored_items if ignored_items else 0
        self.upload_errors = upload_errors if upload_errors else 0
        self.error_codes = set()

    def update_response(self, json):
        assert(self.dataset_id == json.get("dataset_id"), True)

        self.new_items += json.get(NEW_ITEMS)
        self.updated_items += json.get(UPDATED_ITEMS)
        self.ignored_items += json.get(IGNORED_ITEMS)

    def record_error(self, response, num_uploads):
        status = response.status_code
        self.error_codes.add(status)
        self.upload_errors += num_uploads

    def as_dict(self):
        result = {
            "dataset_id": self.dataset_id,
            NEW_ITEMS: self.new_items,
            UPDATED_ITEMS: self.updated_items,
            IGNORED_ITEMS: self.ignored_items,
        }
        if self.error_codes:
            result[ERROR_ITEMS] = self.upload_errors
            result[ERROR_CODES] = self.error_codes
        return result
