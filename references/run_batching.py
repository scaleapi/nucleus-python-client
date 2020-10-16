# Example usage of batching helper_functions
import json
from utils import batch_upload_append
import nucleus

DATA_DIR = "pandaset"


def main():
    client = nucleus.NucleusClient("live_b45ddcc2159a4cadb0fb5ab9b3b0f246")
    response = client.create_dataset({"name": "upload_test"})
    dataset = client.get_dataset(response["dataset_id"])
    file = open(
        DATA_DIR + "/" + "pandaset_items.json", "r"
    )  # open in read mode
    items = json.load(file)

    # Use batching helper!
    batch_upload_append(dataset, items)


if __name__ == "__main__":
    main()
