#!/usr/bin/env python
"""Create a training set from a fixed set of dataset items against a local backend.

Usage:
    python scripts/create_training_set.py --model <model_id> --api-key <key>

The model id and API key are required (training sets are model-scoped). The
backend defaults to the local prod instance at http://localhost:3000.
"""
import argparse

from nucleus import NucleusClient

# The four unique item ids from the request. Note the second token in the
# original prompt was URL-encoded (`di_..b0%2Cdi_..bg`), i.e. two ids joined by
# a comma, so the list below has 4 members, not 3.
ITEM_IDS = [
    "di_d6ng124z6a1006gm86ag",
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
    "di_d6ng124z6a1006gm86c0",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model id to scope/attach the training set to",
    )
    parser.add_argument(
        "--api-key", required=True, help="Scale API key for the target backend"
    )
    parser.add_argument(
        "--name", default="training-set-de-8692", help="Training set name"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:3000/v1/nucleus",
        help="Nucleus API base URL (default: local prod backend)",
    )
    args = parser.parse_args()

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)

    print(
        f"Creating training set {args.name!r} on model {args.model} from {len(ITEM_IDS)} items..."
    )
    training_set = client.create_training_set(
        args.name,
        model=args.model,
        item_ids=ITEM_IDS,
        wait_for_completion=True,
    )

    print(f"\nCreated training set: {training_set.id}")
    print(f"  status:  {getattr(training_set, 'status', '?')}")
    print(
        f"  version: v{training_set.version_major}.{training_set.version_minor}"
    )
    page = training_set.items()
    print(f"  members: {page.total}")
    for item_id in page.item_ids:
        print(f"    - {item_id}")


if __name__ == "__main__":
    main()
