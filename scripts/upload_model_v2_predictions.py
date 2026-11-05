#!/usr/bin/env python
"""Upload run-free ("model v2") predictions to a local Nucleus backend.

Exercises the DE-8678 run-free prediction path: predictions are tied directly
to a Model as ``(model, dataset_item) -> prediction`` with no ModelRun/Dataset.
It fabricates one BoxPrediction per dataset item (targeted by ``dataset_item_id``),
uploads them via ``Model.upload_predictions`` (synchronous), then reads them back
through ``Model.predictions_loc`` to confirm the round-trip.

Usage:
    # create a fresh throwaway model, upload, read back
    python scripts/upload_model_v2_predictions.py --api-key <key>

    # reuse an existing model (prj_* / m_* id)
    python scripts/upload_model_v2_predictions.py --api-key <key> --model <model_id>

The backend defaults to the local prod instance at http://localhost:3000.
"""
import argparse
import time

from nucleus import BoxPrediction, NucleusClient

# The four dataset item ids to attach fake predictions to. (The second token in
# the original request was URL-encoded, i.e. two ids joined by a comma, so this
# list has 4 members.)
ITEM_IDS = [
    "di_d6ng124z6a1006gm86ag",
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
    "di_d6ng124z6a1006gm86c0",
]

LABELS = ["car", "pedestrian", "bicycle", "traffic_light"]


def make_predictions() -> list:
    """One fake BoxPrediction per item, targeted by dataset_item_id."""
    predictions = []
    for i, item_id in enumerate(ITEM_IDS):
        label = LABELS[i % len(LABELS)]
        predictions.append(
            BoxPrediction(
                label=label,
                x=10 + 5 * i,
                y=20 + 5 * i,
                width=50 + 10 * i,
                height=40 + 10 * i,
                confidence=round(0.5 + 0.1 * i, 2),
                metadata={
                    "source": "upload_model_v2_predictions.py",
                    "idx": i,
                },
                # Run-free path: target the item directly by its di_* id.
                dataset_item_id=item_id,
            )
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key", required=True, help="Scale API key for the target backend"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Existing model id to upload to. If omitted, a throwaway model is created.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:3000/v1/nucleus",
        help="Nucleus API base URL (default: local prod backend)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Overwrite existing predictions for the same (item, annotation_id).",
    )
    args = parser.parse_args()

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)

    if args.model:
        model = client.get_model(args.model)
        print(f"Using existing model {model.id} ({model.name!r})")
    else:
        ref = f"model-v2-de8678-{int(time.time())}"
        model = client.create_model(
            name="model-v2 smoke test", reference_id=ref
        )
        print(f"Created model {model.id} (reference_id={ref!r})")

    predictions = make_predictions()
    print(f"\nUploading {len(predictions)} run-free predictions...")
    result = model.upload_predictions(predictions, update=args.update)
    print("upload_predictions ->", result)

    print("\nReading predictions back via predictions_loc:")
    for item_id in ITEM_IDS:
        resp = model.predictions_loc(item_id)
        boxes = resp.get("box", [])
        print(f"  {item_id}: {len(boxes)} box prediction(s)")
        for box in boxes:
            print(
                f"    - label={box.label!r} conf={box.confidence} "
                f"[x={box.x}, y={box.y}, w={box.width}, h={box.height}]"
            )


if __name__ == "__main__":
    main()
