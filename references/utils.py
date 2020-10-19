import json
import tqdm
import nucleus
from nucleus.upload_response import UploadResponse
from nucleus.constants import (
    ITEMS_KEY,
    ANNOTATIONS_KEY,
    ANNOTATIONS_PROCESSED_KEY,
    STATUS_CODE_KEY,
    ERRORS_KEY,
)


def batch_upload_append(args, batch_size=1000):
    """
    Takes large payload, splits it into batches, and calls append sequentially.
    """
    client = nucleus.NucleusClient(args.api_key)
    file = open(args.payload_json_file, "r")  # open in read mode
    payload = json.load(file)
    items = payload[ITEMS_KEY]
    dataset = client.get_dataset(args.dataset_id)

    print("uploading {} items...".format(len(items)))
    batches = [
        items[i : i + batch_size] for i in range(0, len(items), batch_size)
    ]
    agg_response = UploadResponse(json={"dataset_id": args.dataset_id})
    for batch in tqdm.tqdm(batches):
        response = dataset.append({ITEMS_KEY: batch})
        agg_response.update_response(response.json())
    print("upload complete!")
    return agg_response


def batch_upload_annotation(args, batch_size=10):
    """
    Takes large payload, splits it into batches, and calls append sequentially.
    """
    client = nucleus.NucleusClient(args.api_key)
    file = open(args.payload_json_file, "r")  # open in read mode
    payload = json.load(file)
    items = payload[ANNOTATIONS_KEY]
    print("len items is: ", len(items))
    dataset = client.get_dataset(args.dataset_id)
    batches = [
        items[i : i + batch_size] for i in range(0, len(items), batch_size)
    ]
    agg_response = {
        "dataset_id": args.dataset_id,
        ANNOTATIONS_PROCESSED_KEY: 0,
    }
    for batch in tqdm.tqdm(batches):
        response = dataset.annotate({ANNOTATIONS_KEY: batch})
        if STATUS_CODE_KEY in response:
            agg_response[ERRORS_KEY] = response
        else:
            agg_response[ANNOTATIONS_PROCESSED_KEY] += response[
                ANNOTATIONS_PROCESSED_KEY
            ]
    return agg_response
