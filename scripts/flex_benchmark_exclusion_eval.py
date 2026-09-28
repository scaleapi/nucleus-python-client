#!/usr/bin/env python
"""Verify benchmark exclusions actually drop the intended GT *at eval compute*, by scoring the same
model against a control benchmark and an otherwise-identical benchmark that excludes one class.

The trick: evaluate a model with **no predictions**. Then every ground-truth annotation is an
unmatched false negative (TP=0, FP=0, FN=total GT), so the eval's FN rows grouped by `gt_raw_label`
are exactly the benchmark's per-class GT counts — a model-independent oracle for what got scored.

  control  benchmark (no exclusions)                 -> FN by class == full GT makeup
  excluded benchmark (drop "Bomber Aircraft" GT)     -> FN by class == GT makeup minus Bomber

Assertions (no hardcoded totals — derived from the control eval itself):
  1. both evals reach a terminal 'completed' status
  2. TP == 0 and FP == 0 on both (no predictions), so FN == total scored GT
  3. charts.totalCounts.fn == the number of FN example rows (charts vs examples agree)
  4. the excluded class is present in control's FN-by-class and ABSENT in excluded's
  5. every OTHER class has an identical count in both -> only the intended GT was dropped
  6. total_excluded == total_control - control[excluded class]  -> exact drop, nothing extra

Usage:
    # uses $NUCLEUS_PRODCLONE_API_KEY + local prodclone backend by default; restart the backend on
    # the piece-2/3 code first (exclusions are applied at eval compute).
    python scripts/flex_benchmark_exclusion_eval.py

    # keep the throwaway model + benchmarks + evals for inspection
    python scripts/flex_benchmark_exclusion_eval.py --keep
"""
import argparse
import os
import time
import traceback
from collections import Counter
from typing import Dict, List, Optional

from nucleus import LabelExclusionRule, NucleusClient
from nucleus.evaluation_v2 import EvaluationV2

# Scattered prodclone items on ds_d6ng124awzzg0gtc8x50, all with aircraft-class GT.
ITEM_IDS = [
    "di_d6ng124z6a1006gm86ag",
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
    "di_d6ng124z6a1006gm86c0",
    "di_d6ng124z6a1006gm86e0",
    "di_d6ng124z6a1006gm86eg",
]

# The class the "excluded" benchmark drops (annotation-scope, groundTruth).
EXCLUDED_LABEL = "Bomber Aircraft"


class Flex:
    def __init__(self) -> None:
        self.issues: List[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"[ok] {label}")
        else:
            self.issues.append(f"{label}: {detail}")
            print(f"[FAIL] {label}: {detail}")

    def guard(self, label: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            self.issues.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None

    def report(self) -> int:
        print("\n" + "=" * 64)
        if not self.issues:
            print("No issues -- exclusions dropped exactly the intended GT at eval compute.")
            return 0
        print(f"{len(self.issues)} issue(s):")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")
        return 1


def make_ready_benchmark(client, name, exclusion_rules):
    bm = client.create_benchmark(
        name,
        description="exclusion eval-parity flex",
        metadata={"purpose": "flex_benchmark_exclusion_eval.py"},
        item_ids=ITEM_IDS,
        exclusion_rules=exclusion_rules,
    )
    print(
        f"  benchmark {bm.id} status={bm.status!r} items={bm.item_count} "
        f"exclusion_rules={len(bm.exclusion_rules or [])}"
    )
    return bm


def run_eval(client: NucleusClient, benchmark_id: str, model, name: str) -> EvaluationV2:
    print(f"\nEvaluating {name!r} (model has no predictions -> all GT scores as FN)...")
    ev = client.create_benchmark_evaluation_v2(
        benchmark_id=benchmark_id, model_id=model, name=name
    )
    ev.wait_for_completion(timeout_sec=300, poll_interval=3)
    print(f"  eval {ev.id} -> {ev.status}")
    return ev


def fn_by_class(ev: EvaluationV2) -> Dict[str, int]:
    """Count FN example rows grouped by raw GT label (== per-class scored GT for a 0-pred model).

    The examples endpoint caps page size at 100, so page through by offset until we've collected the
    reported total."""
    counts: Counter = Counter()
    offset = 0
    while True:
        page = ev.examples(match_type="FN", limit=100, offset=offset)
        for row in page.rows:
            counts[row.gt_raw_label or "<unlabeled>"] += 1
        offset += len(page.rows)
        if not page.rows or offset >= page.total:
            break
    return dict(counts)


def summarize(ev: EvaluationV2, label: str):
    charts = ev.charts(iou_threshold=0.5)
    tc = charts.totalCounts
    by_class = fn_by_class(ev)
    total_fn_rows = sum(by_class.values())
    print(f"\n{label} ({ev.id}):")
    print(f"  totalCounts: TP={tc.tp} FP={tc.fp} FN={tc.fn}")
    print(f"  FN rows: {total_fn_rows}  by class: {by_class}")
    return tc, by_class, total_fn_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("NUCLEUS_PRODCLONE_API_KEY"))
    parser.add_argument("--endpoint", default="http://localhost:3000/v1/nucleus")
    parser.add_argument("--keep", action="store_true", help="skip cleanup")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("no API key: pass --api-key or set NUCLEUS_PRODCLONE_API_KEY")

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)
    flex = Flex()
    stamp = int(time.time())
    created_benchmarks: List[str] = []
    model = None

    # 0. A fresh model with NO predictions -> every GT scores as FN.
    model = flex.guard(
        "create model",
        lambda: client.create_model(
            name="flex-excl-eval",
            reference_id=f"flex-excl-eval-{stamp}",
            metadata={"purpose": "flex_benchmark_exclusion_eval.py"},
        ),
    )
    if model is None:
        raise SystemExit(flex.report())
    print(f"Created model {model.id}")

    # 1. Control + excluded benchmarks over the SAME items.
    print("\nCreating benchmarks...")
    control = flex.guard(
        "create control benchmark",
        lambda: make_ready_benchmark(client, f"flex-excl-eval-{stamp}-control", None),
    )
    excluded = flex.guard(
        "create excluded benchmark",
        lambda: make_ready_benchmark(
            client,
            f"flex-excl-eval-{stamp}-excluded",
            [LabelExclusionRule(scope="annotation", target="groundTruth", labels=[EXCLUDED_LABEL])],
        ),
    )
    if control is None or excluded is None:
        raise SystemExit(flex.report())
    created_benchmarks += [control.id, excluded.id]

    # 2. Evaluate the same model against each.
    eval_control = flex.guard(
        "eval control",
        lambda: run_eval(client, control.id, model, f"{stamp}-control"),
    )
    eval_excluded = flex.guard(
        "eval excluded",
        lambda: run_eval(client, excluded.id, model, f"{stamp}-excluded"),
    )
    if eval_control is None or eval_excluded is None:
        raise SystemExit(flex.report())

    # 3. Assertions.
    TERMINAL_OK = ("succeeded", "completed")
    flex.check(
        "control eval succeeded",
        eval_control.status in TERMINAL_OK,
        f"status={eval_control.status}",
    )
    flex.check(
        "excluded eval succeeded",
        eval_excluded.status in TERMINAL_OK,
        f"status={eval_excluded.status}",
    )

    tc_c, by_c, fn_rows_c = summarize(eval_control, "CONTROL")
    tc_e, by_e, fn_rows_e = summarize(eval_excluded, "EXCLUDED")

    # No predictions -> TP/FP must be 0 on both, so FN == total scored GT.
    flex.check("control has no TP/FP (0-pred)", tc_c.tp == 0 and tc_c.fp == 0, f"TP={tc_c.tp} FP={tc_c.fp}")
    flex.check("excluded has no TP/FP (0-pred)", tc_e.tp == 0 and tc_e.fp == 0, f"TP={tc_e.tp} FP={tc_e.fp}")

    # charts vs examples agree.
    flex.check(
        "control charts.FN == FN rows",
        tc_c.fn == fn_rows_c,
        f"charts.fn={tc_c.fn} rows={fn_rows_c}",
    )
    flex.check(
        "excluded charts.FN == FN rows",
        tc_e.fn == fn_rows_e,
        f"charts.fn={tc_e.fn} rows={fn_rows_e}",
    )

    # The excluded class is present in control, gone in excluded.
    control_bomber = by_c.get(EXCLUDED_LABEL, 0)
    flex.check(
        f"control includes {EXCLUDED_LABEL!r}",
        control_bomber > 0,
        f"control has no {EXCLUDED_LABEL!r} GT to drop -- pick a class present on these items",
    )
    flex.check(
        f"excluded drops {EXCLUDED_LABEL!r} entirely",
        EXCLUDED_LABEL not in by_e,
        f"excluded still has {by_e.get(EXCLUDED_LABEL)} {EXCLUDED_LABEL!r}",
    )

    # Every OTHER class is untouched -> only the intended GT was dropped.
    other_classes = set(by_c) | set(by_e)
    other_classes.discard(EXCLUDED_LABEL)
    mismatched = {
        cls: (by_c.get(cls, 0), by_e.get(cls, 0))
        for cls in other_classes
        if by_c.get(cls, 0) != by_e.get(cls, 0)
    }
    flex.check(
        "all non-excluded classes unchanged",
        not mismatched,
        f"these classes changed (control,excluded): {mismatched}",
    )

    # Exact drop: excluded total == control total - control[excluded class].
    flex.check(
        "exact GT drop count",
        fn_rows_e == fn_rows_c - control_bomber,
        f"control={fn_rows_c} excluded={fn_rows_e} {EXCLUDED_LABEL!r}={control_bomber} "
        f"(expected excluded={fn_rows_c - control_bomber})",
    )

    print("\n" + "-" * 64)
    print(f"control total GT scored: {fn_rows_c}")
    print(f"excluded total GT scored: {fn_rows_e}  (dropped {fn_rows_c - fn_rows_e})")
    print(f"{EXCLUDED_LABEL!r} GT in control: {control_bomber}")

    if not args.keep:
        print("\nCleaning up...")
        for bm_id in created_benchmarks:
            flex.guard(f"delete benchmark {bm_id}", lambda bm_id=bm_id: client.delete_benchmark(bm_id))
    else:
        print(f"\n--keep: model={model.id} benchmarks={created_benchmarks} "
              f"evals=[{eval_control.id}, {eval_excluded.id}]")

    raise SystemExit(flex.report())


if __name__ == "__main__":
    main()
