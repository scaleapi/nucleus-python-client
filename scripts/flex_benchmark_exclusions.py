#!/usr/bin/env python
"""One big flex of benchmark-owned exclusion rules (create / update / inherit).

The 0.22.4 surface lets a benchmark carry its own exclusion rules (same shape as
the evaluation-V2 rules in :mod:`nucleus.evaluation_v2_exclusions`). They define
the benchmark's *canonical scored content* and are re-applied to every evaluation
computed against the benchmark, so all runs score the same set. Rules can be set
at create time or patched onto a **draft**; a finalized (``"ready"``) benchmark
freezes them, and a version cut inherits the parent's rules unless overridden.

This script walks that surface against a local Nucleus backend:

  1. create_benchmark(draft=True, exclusion_rules=...)  -> draft carries rules
  2. Benchmark.refresh                                  -> read rules back
  3. Benchmark.update(exclusion_rules=[...])            -> replace draft rules
  4. Benchmark.update(exclusion_rules=[])               -> clear draft rules
  5. Benchmark.update(exclusion_rules=[...])            -> re-set before finalize
  6. Benchmark.finalize()                               -> freeze into "ready"
  7. (guard) update rules after finalize                -> must 409; frozen
  8. version cut (parent, omit rules)                   -> inherits parent's rules
  9. version cut (parent, exclusion_rules=[])           -> drops inherited rules
 10. (guard) item-scope rule targeting predictions      -> must 400
 11. (guard) metadata rule with a non-scalar value      -> must 400

The canonical-set guard (steps 10) is the interesting one: an item-scope rule
that drops whole items based on *predictions* would score a different ground-truth
set per model, so the backend rejects it on a benchmark (it stays valid on a
per-eval basis). Prediction-scoped *annotation* rules are fine -- they only drop
predictions, never ground truth -- so step 1 includes one to prove that.

Every step is guarded: a failure is recorded and reported at the end rather than
aborting the run, and the process exits non-zero if anything went wrong.

Usage:
    # uses $NUCLEUS_PRODCLONE_API_KEY and the local prodclone backend by default
    python scripts/flex_benchmark_exclusions.py

    # keep the throwaway benchmarks around instead of deleting them
    python scripts/flex_benchmark_exclusions.py --keep
"""
import argparse
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from nucleus import (
    Benchmark,
    BoxAreaExclusionRule,
    LabelExclusionRule,
    MetadataExclusionRule,
    NucleusAPIError,
    NucleusClient,
)

# --- Fixtures --------------------------------------------------------------
# Same scattered prodclone items the sibling flex scripts use. They live on
# ds_d6ng124awzzg0gtc8x50 and have ground truth (required for benchmark
# membership -- items without GT are skipped).
ITEM_IDS = [
    "di_d6ng124z6a1006gm86ag",
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
    "di_d6ng124z6a1006gm86c0",
    "di_d6ng124z6a1006gm86e0",
    "di_d6ng124z6a1006gm86eg",
]

# Valid benchmark exclusions. Every rule here is legal on a benchmark:
#   - annotation-scope groundTruth label / boxArea rules drop GT content;
#   - a metadata rule drops whole items by a scalar item-metadata value;
#   - a *prediction*-scope ANNOTATION rule is allowed (it only drops predictions,
#     never ground truth) -- this is the counterpart to the rejected item-scope
#     prediction rule in the guard below.
VALID_RULES = [
    LabelExclusionRule(
        scope="annotation", target="groundTruth", labels=["Bomber Aircraft"]
    ),
    BoxAreaExclusionRule(scope="annotation", target="groundTruth", min=1024),
    MetadataExclusionRule(key="is_dark", op="EQ", value=True),
    LabelExclusionRule(
        scope="annotation", target="prediction", labels=["ignore"]
    ),
]

# A different, smaller rule set used to prove update_benchmark replaces (not
# merges) a draft's exclusions.
REPLACEMENT_RULES = [
    LabelExclusionRule(
        scope="annotation", target="groundTruth", labels=["Fighter Aircraft"]
    ),
    MetadataExclusionRule(key="difficulty", op="IN", value=["hard", "extreme"]),
]


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

    def expect_raises(self, label: str, exc_type, fn):
        """Invert a step: the call SHOULD raise exc_type. Records an issue if
        it doesn't, or raises the wrong type."""
        print(f"\n=== {label} (expect {exc_type.__name__}) ===")
        try:
            fn()
        except exc_type as exc:
            print(f"[ok] {label}: raised {type(exc).__name__} as expected")
            print(f"  {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self.issues.append(
                f"{label}: raised {type(exc).__name__} "
                f"(wanted {exc_type.__name__}): {exc}"
            )
            print(
                f"[FAIL] {label}: wrong exception {type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return
        self.issues.append(f"{label}: did NOT raise {exc_type.__name__}")
        print(f"[FAIL] {label}: no exception raised")

    def report(self) -> int:
        print("\n" + "=" * 60)
        if not self.issues:
            print("No issues -- the benchmark-exclusion flex completed cleanly.")
            return 0
        print(f"{len(self.issues)} issue(s) encountered:")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")
        return 1


def rule_count(bm: Benchmark) -> int:
    return len(bm.exclusion_rules or [])


def rule_signatures(rules: Optional[List[Dict[str, Any]]]) -> List[Tuple]:
    """Order-independent (type, scope, target) signature of a rule set, for
    comparing what the server stored against what we sent. The server may
    reorder / normalize, so compare by signature rather than exact equality."""
    sigs = []
    for r in rules or []:
        sigs.append((r.get("type"), r.get("scope"), r.get("target")))
    return sorted(sigs, key=lambda t: tuple("" if x is None else str(x) for x in t))


def show(label: str, bm: Benchmark) -> None:
    print(
        f"  {label}: {bm.id} status={bm.status!r} items={bm.item_count} "
        f"exclusion_rules={rule_count(bm)}"
    )
    for r in bm.exclusion_rules or []:
        print(f"      - {r}")


def assert_status(bm: Benchmark, expected: str) -> None:
    if bm.status != expected:
        raise RuntimeError(
            f"expected status={expected!r}, got {bm.status!r} on {bm.id}"
        )


def assert_rule_count(bm: Benchmark, expected: int) -> None:
    actual = rule_count(bm)
    if actual != expected:
        raise RuntimeError(
            f"expected {expected} exclusion rule(s) on {bm.id}, got {actual}: "
            f"{bm.exclusion_rules!r}"
        )


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
        "--keep",
        action="store_true",
        help="Skip the final delete_benchmark cleanup.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "no API key: pass --api-key or set NUCLEUS_PRODCLONE_API_KEY"
        )

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)
    flex = Flex()
    stamp = int(time.time())
    created: List[str] = []

    def track(bm: Optional[Benchmark]) -> Optional[Benchmark]:
        if bm is not None:
            created.append(bm.id)
        return bm

    # 1. Create a mutable draft carrying the full valid rule set.
    benchmark: Optional[Benchmark] = track(
        flex.step(
            "1. create_benchmark (draft, with exclusion_rules)",
            lambda: client.create_benchmark(
                f"flex-bm-excl-{stamp}",
                description="benchmark exclusion rules flex",
                metadata={"purpose": "flex_benchmark_exclusions.py"},
                item_ids=ITEM_IDS,
                draft=True,
                exclusion_rules=VALID_RULES,
            ),
        )
    )
    if benchmark is None:
        raise SystemExit(flex.report())
    show("draft", benchmark)

    def _check_created() -> None:
        assert_status(benchmark, "draft")
        assert_rule_count(benchmark, len(VALID_RULES))

    flex.step("1b. draft is 'draft' and carries all rules", _check_created)

    # 2. Round-trip: refresh and confirm the stored signatures match what we sent.
    def _read_back() -> None:
        benchmark.refresh()
        show("refreshed", benchmark)
        want = rule_signatures([r.to_api_dict() for r in VALID_RULES])
        got = rule_signatures(benchmark.exclusion_rules)
        if want != got:
            raise RuntimeError(
                f"stored rule signatures differ.\n  sent: {want}\n  got:  {got}"
            )

    flex.step("2. refresh + verify stored rule signatures", _read_back)

    # 3. Replace the draft's rules with a different, smaller set.
    def _replace() -> None:
        show(
            "after replace",
            benchmark.update(exclusion_rules=REPLACEMENT_RULES),
        )
        assert_rule_count(benchmark, len(REPLACEMENT_RULES))
        want = rule_signatures([r.to_api_dict() for r in REPLACEMENT_RULES])
        got = rule_signatures(benchmark.exclusion_rules)
        if want != got:
            raise RuntimeError(
                f"replacement not applied.\n  sent: {want}\n  got:  {got}"
            )

    flex.step("3. update_benchmark (replace exclusion_rules)", _replace)

    # 4. Clear the draft's rules with an explicit empty list.
    def _clear() -> None:
        show("after clear", benchmark.update(exclusion_rules=[]))
        assert_rule_count(benchmark, 0)

    flex.step("4. update_benchmark (clear via [])", _clear)

    # 5. Re-set the full rule set so the finalized benchmark carries exclusions.
    def _reset() -> None:
        show("after re-set", benchmark.update(exclusion_rules=VALID_RULES))
        assert_rule_count(benchmark, len(VALID_RULES))

    flex.step("5. update_benchmark (re-set before finalize)", _reset)

    # 6. Freeze the draft. Exclusions become immutable.
    def _finalize() -> None:
        show("finalized", benchmark.finalize())
        assert_status(benchmark, "ready")
        assert_rule_count(benchmark, len(VALID_RULES))

    flex.step("6. finalize_benchmark", _finalize)

    # 7. A ready benchmark rejects exclusion changes (server 409).
    flex.expect_raises(
        "7. update exclusion_rules after finalize (must fail)",
        NucleusAPIError,
        lambda: benchmark.update(
            exclusion_rules=[
                LabelExclusionRule(
                    scope="annotation", target="groundTruth", labels=["x"]
                )
            ]
        ),
    )

    # 8. Version cut: omit exclusion_rules -> inherit the parent's rules.
    def _cut_inherit() -> None:
        child = track(
            client.create_benchmark(
                f"flex-bm-excl-{stamp}-v2-inherit",
                description="version cut inheriting exclusions",
                item_ids=ITEM_IDS,
                draft=True,
                parent_benchmark_id=benchmark.id,
                # exclusion_rules omitted -> inherit parent's
            )
        )
        show("child (inherited)", child)
        want = rule_signatures(benchmark.exclusion_rules)
        got = rule_signatures(child.exclusion_rules)
        if want != got:
            raise RuntimeError(
                f"child did not inherit parent's exclusions.\n"
                f"  parent: {want}\n  child:  {got}"
            )

    flex.step("8. version cut inherits parent's exclusions", _cut_inherit)

    # 9. Version cut: exclusion_rules=[] -> drop the inherited rules.
    def _cut_clear() -> None:
        child = track(
            client.create_benchmark(
                f"flex-bm-excl-{stamp}-v2-clear",
                description="version cut dropping inherited exclusions",
                item_ids=ITEM_IDS,
                draft=True,
                parent_benchmark_id=benchmark.id,
                exclusion_rules=[],  # explicit clear
            )
        )
        show("child (cleared)", child)
        assert_rule_count(child, 0)

    flex.step("9. version cut with [] drops inherited exclusions", _cut_clear)

    # 10. Guard: an item-scope rule targeting predictions is rejected on a
    #     benchmark (it would score a different item set per model).
    flex.expect_raises(
        "10. create with item-scope prediction rule (must 400)",
        NucleusAPIError,
        lambda: track(
            client.create_benchmark(
                f"flex-bm-excl-{stamp}-bad-pred",
                item_ids=ITEM_IDS,
                draft=True,
                exclusion_rules=[
                    LabelExclusionRule(
                        scope="item", target="prediction", labels=["ignore"]
                    )
                ],
            )
        ),
    )

    # 11. Guard: a metadata rule with a non-scalar value is rejected (Postgres
    #     compares metadata as text, so an object value never matches).
    flex.expect_raises(
        "11. create with non-scalar metadata value (must 400)",
        NucleusAPIError,
        lambda: track(
            client.create_benchmark(
                f"flex-bm-excl-{stamp}-bad-meta",
                item_ids=ITEM_IDS,
                draft=True,
                exclusion_rules=[
                    MetadataExclusionRule(
                        key="tags", op="EQ", value={"nested": "object"}
                    )
                ],
            )
        ),
    )

    if not args.keep:
        for bm_id in created:
            flex.step(
                f"cleanup: delete_benchmark {bm_id}",
                lambda bm_id=bm_id: client.delete_benchmark(bm_id),
            )
    else:
        print(f"\n--keep: leaving {len(created)} benchmark(s) in place: {created}")

    print("\nDone.")
    print(f"  primary benchmark: {benchmark.id} (status={benchmark.status})")
    print(f"  exclusion_rules:   {rule_count(benchmark)}")
    raise SystemExit(flex.report())


if __name__ == "__main__":
    main()
