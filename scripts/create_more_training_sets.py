#!/usr/bin/env python
"""Create a child version + an unversioned "mix" training set (DE-8692 manual test).

Usage:
    poetry run python scripts/create_more_training_sets.py \
        --model <model_id> --api-key <key> --parent <trs_id>

- Child version: downstream of --parent, prunes one inherited item and adds one new one.
- Mix: unversioned root, unioning the parent set + the child set (+ one extra item).
"""
import argparse

from nucleus import NucleusClient

# Item chosen to prune from the parent, and items to add to the child version.
REMOVE_FROM_PARENT = "di_d6ng124z6a1006gm86c0"
ADD_TO_CHILD = ["di_d6ng124z6a1006gm86eg"]
# Extra item folded into the unversioned "mix" set alongside the two source sets.
EXTRA_FOR_MIX = ["di_d6ng124z6a1006gm86e0"]


def summarize(client: NucleusClient, ts) -> None:
    page = client.list_training_set_items(ts.id, limit=1000)
    print(
        f"  {ts.id}  v{ts.version_major}.{ts.version_minor}  ({page.total} items)"
    )
    for item_id in page.item_ids:
        print(f"    - {item_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument(
        "--parent", required=True, help="Parent training set id (trs_*)"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:3000/v1/nucleus"
    )
    args = parser.parse_args()

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)

    print(
        f"Creating child version of {args.parent} "
        f"(prune {REMOVE_FROM_PARENT}, add {ADD_TO_CHILD})..."
    )
    # Downstream version: inherits the parent's members, adds item_ids, prunes removed_item_ids —
    # final set = parent ∪ added ∖ removed, in one call.
    child = client.create_training_set(
        "training-set-de-8692-child",
        model=args.model,
        parent_training_set_id=args.parent,
        item_ids=ADD_TO_CHILD,
        removed_item_ids=[REMOVE_FROM_PARENT],
        wait_for_completion=True,
    )
    print("Child version:")
    summarize(client, child)

    print(
        f"\nCreating unversioned mix of {args.parent} + {child.id} (+ {EXTRA_FOR_MIX})..."
    )
    mix = client.create_training_set(
        "training-set-de-8692-mix",
        model=args.model,
        training_set_ids=[args.parent, child.id],
        item_ids=EXTRA_FOR_MIX,
        wait_for_completion=True,
    )
    print("Mix set:")
    summarize(client, mix)


if __name__ == "__main__":
    main()
