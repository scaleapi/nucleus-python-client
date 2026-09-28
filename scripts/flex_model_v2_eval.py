#!/usr/bin/env python
"""End-to-end flex of the DE-8678 run-free ("model v2") prediction + eval path.

Exercises, against a local Nucleus backend, the full new surface in one run:

  1. create_model                     -> a fresh throwaway Model
  2. Model.upload_predictions         -> run-free preds tied to (model, item)
  3. Model.predictions_loc            -> read them back
  4. create_benchmark_evaluation_v2   -> eval #1, anchored on model_id (not a run)
  5. EvaluationV2.wait_for_completion / .charts / .examples  -> results
  6. Model.copy_predictions_from_run  -> backfill a real v1 run's preds onto the model
  7. create_benchmark_evaluation_v2   -> eval #2 (now includes the copied run)
  8. charts/examples again            -> compare scores before vs. after the copy

The two-eval design makes the copy's effect visible: eval #1 scores only the
fabricated boxes uploaded in step 2; eval #2 also scores the real predictions
copied from the source run in step 6.

Usage:
    # uses $NUCLEUS_PRODCLONE_API_KEY and the local prodclone backend by default
    python scripts/flex_model_v2_eval.py

    # explicit key / endpoint
    python scripts/flex_model_v2_eval.py --api-key <key> \
        --endpoint http://localhost:3000/v1/nucleus
"""
import argparse
import os
import time
from typing import List

from nucleus import BoxPrediction, NucleusClient
from nucleus.evaluation_v2 import EvaluationV2

# --- Fixtures discovered on the local prodclone server ----------------------
# "Super Small Benchmark": 3 items on ds_d6ng124awzzg0gtc8x50.
BENCHMARK_ID = "bm_d8xgaf1zc4110htzjvq0"
# The two item ids from the request (the benchmark has a 3rd we leave uncovered
# on purpose -- it scores as a false negative, which is the point of the design).
ITEM_IDS = [
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
]
# Existing model prj_d9akaq2zc4111cx32qm0 ("SAM3")'s source v1 run. Its copied
# predictions on the server carry annotation ids prefixed with this run id.
SOURCE_RUN_ID = "run_da2dw00zc41599nre630"

LABELS = ["car", "pedestrian"]


def make_predictions() -> List[BoxPrediction]:
    """One fabricated BoxPrediction per item, targeted by dataset_item_id.

    Only box/polygon/cuboid predictions are accepted on the run-free path.
    """
    predictions = []
    for i, item_id in enumerate(ITEM_IDS):
        predictions.append(
            BoxPrediction(
                label=LABELS[i % len(LABELS)],
                x=10 + 5 * i,
                y=20 + 5 * i,
                width=50 + 10 * i,
                height=40 + 10 * i,
                confidence=round(0.5 + 0.1 * i, 2),
                metadata={"source": "flex_model_v2_eval.py", "idx": i},
                dataset_item_id=item_id,  # run-free: target the item directly
            )
        )
    return predictions


def read_back(model, header: str) -> None:
    """Round-trip the model's run-free predictions via predictions_loc."""
    print(f"\n{header}")
    for item_id in ITEM_IDS:
        resp = model.predictions_loc(item_id)
        counts = {k: len(v) for k, v in resp.items() if v}
        total = sum(counts.values())
        print(f"  {item_id}: {total} prediction(s) {counts or ''}")


def run_eval(client: NucleusClient, model, name: str) -> EvaluationV2:
    """Kick off a benchmark eval anchored on the model (run-free), then wait."""
    print(f"\nCreating benchmark evaluation {name!r} (model-anchored)...")
    evaluation = client.create_benchmark_evaluation_v2(
        benchmark_id=BENCHMARK_ID,
        model_id=model,  # accepts a Model or a prj_* id; NOT a model run
        name=name,
    )
    print(f"  eval id={evaluation.id} status={evaluation.status}")
    evaluation.wait_for_completion(timeout_sec=300, poll_interval=3)
    print(f"  -> terminal status={evaluation.status}")
    return evaluation


def print_results(evaluation: EvaluationV2) -> None:
    """Summarize charts + example match-type breakdown for an eval."""
    charts = evaluation.charts(iou_threshold=0.5)
    tc = charts.totalCounts
    m = charts.mapSummary
    print(
        f"  totals: TP={tc.tp} FP={tc.fp} FN={tc.fn} "
        f"(preds w/ confidence={tc.predsWithConfidence})"
    )
    print(f"  mAP@50={m.mapAt50} mAP@75={m.mapAt75} mAP@50-95={m.mapAt5095}")
    if charts.perClassAp:
        top = sorted(charts.perClassAp, key=lambda c: c.ap, reverse=True)[:5]
        print(
            "  per-class AP (top 5): "
            + ", ".join(f"{c.classLabel}={c.ap:.3f}" for c in top)
        )

    page = evaluation.examples(limit=100)
    by_type: dict = {}
    for row in page.rows:
        by_type[row.match_type] = by_type.get(row.match_type, 0) + 1
    print(f"  examples: total={page.total} by_match_type={by_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NUCLEUS_PRODCLONE_API_KEY"),
        help="Scale API key (default: $NUCLEUS_PRODCLONE_API_KEY).",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:3000/v1/nucleus",
        help="Nucleus API base URL (default: local prodclone backend).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Overwrite existing predictions for the same (item, annotation_id).",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "no API key: pass --api-key or set NUCLEUS_PRODCLONE_API_KEY"
        )

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)

    # 1. Create a fresh throwaway model.
    ref = f"flex-model-v2-eval-{int(time.time())}"
    model = client.create_model(
        name="flex model-v2 eval",
        reference_id=ref,
        metadata={"purpose": "DE-8678 run-free eval flex"},
    )
    print(f"Created model {model.id} (reference_id={ref!r})")

    # 2 + 3. Upload run-free predictions, then read them back.
    predictions = make_predictions()
    print(f"\nUploading {len(predictions)} run-free predictions...")
    print(
        "  upload_predictions ->",
        model.upload_predictions(predictions, update=args.update),
    )
    read_back(model, "predictions_loc after upload:")

    # 4 + 5. Eval #1 -- scores only the fabricated boxes above.
    eval1 = run_eval(client, model, f"{ref}-eval1-uploaded-only")
    print("\n=== Eval #1 results (uploaded predictions only) ===")
    print_results(eval1)

    # 6. Copy a real v1 run's predictions onto the model (run-free backfill).
    print(f"\nCopying predictions from source run {SOURCE_RUN_ID}...")
    print(
        "  copy_predictions_from_run ->",
        model.copy_predictions_from_run(SOURCE_RUN_ID),
    )
    read_back(model, "predictions_loc after copy:")

    # 7 + 8. Eval #2 -- now includes the copied run's predictions.
    eval2 = run_eval(client, model, f"{ref}-eval2-with-copied-run")
    print("\n=== Eval #2 results (uploaded + copied run) ===")
    print_results(eval2)

    print("\nDone.")
    print(f"  model:  {model.id}")
    print(f"  eval1:  {eval1.id}")
    print(f"  eval2:  {eval2.id}")


if __name__ == "__main__":
    main()
