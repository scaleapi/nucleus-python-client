#!/usr/bin/env python
"""One big flex of the run-free ("model v2") prediction setup (DE-8678).

Ties together everything the smaller scripts in this directory exercise
piecemeal -- create_training_set.py, upload_model_v2_predictions.py, and
flex_model_v2_eval.py -- into a single end-to-end run against a local Nucleus
backend:

  1. create_model                 -> a fresh throwaway destination Model
  2. Model.upload_predictions     -> run-free preds across items scattered
                                     across datasets (targeted by di_* id)
  3. Model.predictions_loc        -> read them back
  4. resolve a SOURCE run         -> either --source-run, or the first run of
                                     --source-model via Model.model_runs()
  5. Model.copy_predictions_from_run
                                  -> copy that other model's run into THIS
                                     model's run-free prediction set
  6. Model.predictions_loc        -> read back the merged (uploaded + copied) set
  7. (optional, --eval) create_benchmark_evaluation_v2 anchored on model_id,
     then charts/examples -- proves the merged predictions score.

Every step is guarded: a failure is recorded and reported at the end rather
than aborting the run, and the process exits non-zero if anything went wrong --
so you can see *all* the issues in one pass, which is the point of the flex.

Usage:
    # uses $NUCLEUS_PRODCLONE_API_KEY and the local prodclone backend by default;
    # copies the default SAM3 source run into the new model.
    python scripts/flex_model_run_free.py

    # discover the source run from a source model instead of hardcoding it
    python scripts/flex_model_run_free.py --source-model prj_d9akaq2zc4111cx32qm0

    # copy an explicit run, and also run a benchmark eval at the end
    python scripts/flex_model_run_free.py --source-run run_da2dw00zc41599nre630 --eval
"""
import argparse
import os
import time
import traceback
from typing import List, Optional

from nucleus import BoxPrediction, Model, NucleusClient
from nucleus.evaluation_v2 import EvaluationV2

# --- Fixtures --------------------------------------------------------------
# The union of every dataset item id the sibling scripts touch. These are
# "scattered about" -- run-free predictions are (model, item) scoped, so a
# single model can hold predictions for items regardless of which dataset they
# live in. Targeted by di_* id (no reference_id / dataset needed).
ITEM_IDS = [
    "di_d6ng124z6a1006gm86ag",
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
    "di_d6ng124z6a1006gm86c0",
    "di_d6ng124z6a1006gm86e0",
    "di_d6ng124z6a1006gm86eg",
]

LABELS = ["car", "pedestrian", "bicycle", "traffic_light"]

# Defaults for step 4, borrowed from flex_model_v2_eval.py: the existing "SAM3"
# model and one of its legacy v1 runs. --source-model / --source-run override.
DEFAULT_SOURCE_MODEL = "prj_d9akaq2zc4111cx32qm0"
DEFAULT_SOURCE_RUN = "run_da2dw00zc41599nre630"

# "Super Small Benchmark" (3 items) -- only used with --eval.
BENCHMARK_ID = "bm_d8xgaf1zc4110htzjvq0"


class Flex:
    """Runs each stage guarded, collecting issues for a final report."""

    def __init__(self) -> None:
        self.issues: List[str] = []

    def step(self, label: str, fn):
        """Run fn(), print an OK/FAIL line, record any exception. Returns
        fn()'s result, or None on failure."""
        print(f"\n=== {label} ===")
        try:
            result = fn()
            print(f"[ok] {label}")
            return result
        except Exception as exc:  # noqa: BLE001 -- we want to surface everything
            self.issues.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None

    def report(self) -> int:
        print("\n" + "=" * 60)
        if not self.issues:
            print("No issues -- the run-free flex completed cleanly.")
            return 0
        print(f"{len(self.issues)} issue(s) encountered:")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")
        return 1


def make_predictions() -> List[BoxPrediction]:
    """One fabricated BoxPrediction per scattered item (targeted by di_* id).

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
                confidence=round(0.5 + 0.05 * i, 2),
                metadata={"source": "flex_model_run_free.py", "idx": i},
                dataset_item_id=item_id,  # run-free: target the item directly
            )
        )
    return predictions


def read_back(model: Model, header: str) -> int:
    """Round-trip the model's run-free predictions; return the total count."""
    print(header)
    total = 0
    for item_id in ITEM_IDS:
        resp = model.predictions_loc(item_id)
        counts = {k: len(v) for k, v in resp.items() if v}
        n = sum(counts.values())
        total += n
        print(f"  {item_id}: {n} prediction(s) {counts or ''}")
    print(f"  total across items: {total}")
    return total


def resolve_source_run(
    client: NucleusClient,
    source_run: Optional[str],
    source_model: Optional[str],
) -> str:
    """Pick the run to copy. Explicit --source-run wins; otherwise discover the
    first run of --source-model via Model.model_runs()."""
    if source_run:
        print(f"  using explicit source run {source_run}")
        return source_run
    model = client.get_model(source_model)
    runs = model.model_runs()
    print(f"  source model {model.id} ({model.name!r}) has {len(runs)} run(s)")
    if not runs:
        raise RuntimeError(
            f"source model {source_model} has no runs to copy from"
        )
    chosen = runs[0]
    print(f"  chosen source run: {chosen}")
    return chosen


def run_eval(client: NucleusClient, model: Model, name: str) -> EvaluationV2:
    print(f"Creating benchmark evaluation {name!r} (model-anchored)...")
    evaluation = client.create_benchmark_evaluation_v2(
        benchmark_id=BENCHMARK_ID,
        model_id=model,  # run-free anchor -- a Model or prj_* id, NOT a run
        name=name,
    )
    print(f"  eval id={evaluation.id} status={evaluation.status}")
    evaluation.wait_for_completion(timeout_sec=300, poll_interval=3)
    print(f"  -> terminal status={evaluation.status}")
    charts = evaluation.charts(iou_threshold=0.5)
    tc = charts.totalCounts
    m = charts.mapSummary
    print(f"  totals: TP={tc.tp} FP={tc.fp} FN={tc.fn}")
    print(f"  mAP@50={m.mapAt50} mAP@50-95={m.mapAt5095}")
    return evaluation


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
        "--source-run",
        default=None,
        help=f"Run id (run_*) to copy into the new model. Default: discover "
        f"from --source-model, falling back to {DEFAULT_SOURCE_RUN}.",
    )
    parser.add_argument(
        "--source-model",
        default=DEFAULT_SOURCE_MODEL,
        help="Model id (prj_*) whose first run is copied when --source-run is "
        f"omitted. Default: {DEFAULT_SOURCE_MODEL} (SAM3).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Overwrite existing predictions for the same (item, annotation_id).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Also run a benchmark eval on the merged predictions at the end.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("no API key: pass --api-key or set NUCLEUS_PRODCLONE_API_KEY")
    # If the user pinned a source run explicitly, don't also try to discover one.
    default_source_run = None if args.source_model != DEFAULT_SOURCE_MODEL else DEFAULT_SOURCE_RUN
    source_run = args.source_run or default_source_run

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)
    flex = Flex()

    # 1. Create a fresh throwaway destination model.
    ref = f"flex-run-free-{int(time.time())}"
    model = flex.step(
        "1. create_model",
        lambda: client.create_model(
            name="run-free flex",
            reference_id=ref,
            metadata={"purpose": "DE-8678 run-free end-to-end flex"},
        ),
    )
    if model is None:
        # Nothing downstream can run without a model; report and bail.
        raise SystemExit(flex.report())
    print(f"  model {model.id} (reference_id={ref!r})")

    # 2. Upload run-free predictions across the scattered items.
    predictions = make_predictions()
    flex.step(
        f"2. upload_predictions ({len(predictions)} preds, scattered items)",
        lambda: print("  ->", model.upload_predictions(predictions, update=args.update)),
    )

    # 3. Read them back.
    flex.step(
        "3. predictions_loc (after upload)",
        lambda: read_back(model, "  uploaded predictions:"),
    )

    # 4. Resolve + copy a run from another model into this model's pred set.
    resolved = flex.step(
        "4. resolve + copy_predictions_from_run (one model -> this model)",
        lambda: (
            lambda run: print(
                "  ->", model.copy_predictions_from_run(run)
            )
        )(resolve_source_run(client, source_run, args.source_model)),
    )
    _ = resolved

    # 5. Read back the merged set (uploaded + copied).
    flex.step(
        "5. predictions_loc (after copy -- merged set)",
        lambda: read_back(model, "  merged predictions:"),
    )

    # 6. Optional eval on the merged predictions.
    if args.eval:
        flex.step(
            "6. create_benchmark_evaluation_v2 (merged preds)",
            lambda: run_eval(client, model, f"{ref}-eval-merged"),
        )

    print("\nDone.")
    print(f"  model: {model.id}")
    raise SystemExit(flex.report())


if __name__ == "__main__":
    main()
