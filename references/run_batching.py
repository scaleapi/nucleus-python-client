# Example usage of batching helper_functions
import argparse
from utils import batch_upload_append, batch_upload_annotation

commands_dict = {
    "append": batch_upload_append,
    "annotate": batch_upload_annotation,
}


def main(args):
    # Use batching helper!
    if args.action not in commands_dict:
        raise Exception(
            "invalid argument", "Invalid Command specified by action flag"
        )
    command = commands_dict[args.action]
    response = command(args)
    print(response.json())


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
        "--payload_json_file",
        type=str,
        required=True,
        help="Path to file containing JSON payload",
    )
    main_args = parser.parse_args()
    print(main_args)
    main(main_args)
