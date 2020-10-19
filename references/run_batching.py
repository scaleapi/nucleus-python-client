# Example usage of batching helper_functions
import json
import argparse
from utils import (
    batch_upload_append,
    batch_upload_annotation,
    batch_upload_prediction,
)

APPEND_ACTION = "append"
ANNOTATE_ACTION = "annotate"
PREDICT_ACTION = "predict"

commands_dict = {
    APPEND_ACTION: batch_upload_append,
    ANNOTATE_ACTION: batch_upload_annotation,
    PREDICT_ACTION: batch_upload_prediction,
}


def main(args):
    # Use batching helper!
    if args.action not in commands_dict:
        raise Exception(
            "invalid Argument", "Invalid Command specified by action flag"
        )
    # Check if model run id is a required param
    if args.action == PREDICT_ACTION and not args.model_run_id:
        raise Exception(
            "invalid Argument",
            "Invalid Arguments for action:{}. Must specify model_run_id".format(
                PREDICT_ACTION
            ),
        )
    command = commands_dict[args.action]
    api_key = args.api_key
    batch_size = args.batch_size
    dataset_id = args.dataset_id
    # Load payload from JSON file
    file = open(args.payload_json_file, "r")  # open in read mode
    payload = json.load(file)
    if args.action == PREDICT_ACTION:
        model_run_id = args.model_run_id
        response = command(
            api_key, dataset_id, model_run_id, payload, batch_size
        )
    else:
        response = command(api_key, dataset_id, payload, batch_size)
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        help="Which dataset operation to perform (append, annotate)",
    )
    parser.add_argument(
        "--api_key", type=str, required=True, help="Scale API key"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of items to upload per request",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="ID of Nucleus dataset being updated.",
    )
    parser.add_argument(
        "--model_run_id",
        type=str,
        help="ID of Model Run being updated. Only required for prediction.",
    )
    parser.add_argument(
        "--payload_json_file",
        type=str,
        required=True,
        help="Path to file containing JSON payload",
    )
    main_args = parser.parse_args()
    print(main_args)
    main(main_args)
