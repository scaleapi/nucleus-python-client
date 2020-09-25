NEW_ITEMS = "new_items"
UPDATED_ITEMS = "updated_items"
IGNORED_ITEMS = "ignored_items"
ERROR_ITEMS = "errors"

class UploadResponse:

    def __init__(self, json={}):
        dataset_id = json.get("dataset_id")
        new_items = json.get(NEW_ITEMS)
        updated_items = json.get(UPDATED_ITEMS)
        ignored_items = json.get(IGNORED_ITEMS)

        self.dataset_id = dataset_id
        self.new_items = new_items if new_items else 0
        self.updated_items = updated_items if updated_items else 0
        self.ignored_items = ignored_items if ignored_items else 0
        self.errors = {}

    def update_response(self, json):
        assert(self.dataset_id == json.get("dataset_id"), True)

        self.new_items += json.get(NEW_ITEMS)
        self.updated_items += json.get(UPDATED_ITEMS)
        self.ignored_items += json.get(IGNORED_ITEMS)

    def record_error(self, response):
        key = f"HTTP Status: {response.status_code}"
        if (key in self.errors):
            self.errors[key]+= 1
        else:
            self.errors[key] = 1

    def as_dict(self):
        return {
            "dataset_id": self.dataset_id,
            NEW_ITEMS: self.new_items,
            UPDATED_ITEMS: self.updated_items,
            IGNORED_ITEMS: self.ignored_items,
            ERROR_ITEMS: self.errors,
        }
