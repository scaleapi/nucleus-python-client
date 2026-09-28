#!/usr/bin/env python
"""One big flex of benchmark rollup taxonomy (create / update / finalize).

The 0.22.3 surface lets a benchmark carry a class taxonomy. Taxonomy can be
set at create time or patched onto a **draft**; a finalized (``"ready"``)
benchmark freezes it. This script walks the incremental draft path against a
local Nucleus backend:

  1. create_benchmark(draft=True, item_ids=...)  -> empty-taxonomy draft
  2. Benchmark.update(rollup_groups=...)         -> attach the rollup
  3. Benchmark.refresh / items()                 -> read taxonomy + membership
  4. Benchmark.finalize()                        -> freeze into ``"ready"``
  5. (guard) update rollup after finalize        -> must 409; taxonomy is frozen

Every step is guarded: a failure is recorded and reported at the end rather
than aborting the run, and the process exits non-zero if anything went wrong.

Usage:
    # uses $NUCLEUS_PRODCLONE_API_KEY and the local prodclone backend by default
    python scripts/flex_benchmark_rollups.py

    # keep the throwaway benchmark around instead of deleting it
    python scripts/flex_benchmark_rollups.py --keep
"""
import argparse
import os
import time
import traceback
from typing import List, Optional

from nucleus import Benchmark, NucleusAPIError, NucleusClient, RollupGroup

# --- Fixtures --------------------------------------------------------------
# Same scattered prodclone items the sibling flex scripts use. They live on
# ds_d6ng124awzzg0gtc8x50 and have ground truth (required for benchmark
# membership — items without GT are skipped).
ITEM_IDS = [
    "di_d6ng124z6a1006gm86ag",
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
    "di_d6ng124z6a1006gm86c0",
    "di_d6ng124z6a1006gm86e0",
    "di_d6ng124z6a1006gm86eg",
]

# Actual GT classes on this benchmark, rolled into canonical eval classes.
# A label may appear in at most one group.
ROLLUP_GROUPS = [
    RollupGroup(
        "Combat Aircraft",
        ["Attack Aircraft", "Bomber Aircraft", "Fighter Aircraft"],
    ),
    RollupGroup(
        "C2 / ISR / Patrol Aircraft",
        ["Command Control - Reconn - Patrol Aircraft"],
    ),
    RollupGroup("Helicopter", ["Attack/Transport Helicopter"]),
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
            print("No issues -- the benchmark-rollup flex completed cleanly.")
            return 0
        print(f"{len(self.issues)} issue(s) encountered:")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")
        return 1


def show(label: str, bm: Benchmark) -> None:
    print(
        f"  {label}: {bm.id} status={bm.status!r} items={bm.item_count} "
        f"taxonomy={bm.allowed_label_matches_id} "
        f"class_agnostic={bm.class_agnostic}"
    )


def assert_status(bm: Benchmark, expected: str) -> None:
    if bm.status != expected:
        raise RuntimeError(
            f"expected status={expected!r}, got {bm.status!r} on {bm.id}"
        )


def assert_untyped(bm: Benchmark) -> None:
    if bm.allowed_label_matches_id:
        raise RuntimeError(
            f"draft already has a taxonomy: {bm.allowed_label_matches_id}"
        )


def assert_has_taxonomy(bm: Benchmark) -> None:
    """A typed benchmark surfaces an allowed_label_matches_id after the
    server materializes the inline rollup into a config."""
    if not bm.allowed_label_matches_id:
        raise RuntimeError(
            f"expected a taxonomy (allowed_label_matches_id) on {bm.id}, "
            f"got {bm.allowed_label_matches_id!r} (class_agnostic="
            f"{bm.class_agnostic})"
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
    name = f"flex-bm-rollup-{int(time.time())}"

    # 1. Create a mutable draft with members, no taxonomy yet.
    benchmark: Optional[Benchmark] = flex.step(
        "1. create_benchmark (draft, no taxonomy)",
        lambda: client.create_benchmark(
            name,
            description="benchmark rollup taxonomy flex",
            metadata={"purpose": "flex_benchmark_rollups.py"},
            item_ids=ITEM_IDS,
            draft=True,
        ),
    )
    if benchmark is None:
        raise SystemExit(flex.report())
    show("draft", benchmark)

    def _check_draft() -> None:
        assert_status(benchmark, "draft")
        assert_untyped(benchmark)

    flex.step("1b. draft status is 'draft' and untyped", _check_draft)

    # 2. Attach the rollup while the benchmark is still a draft.
    def _add_rollup() -> None:
        show("after update", benchmark.update(rollup_groups=ROLLUP_GROUPS))
        assert_has_taxonomy(benchmark)

    flex.step("2. update_benchmark (add rollup_groups)", _add_rollup)

    # 3. Round-trip: refresh + list members.
    def _read_back() -> None:
        benchmark.refresh()
        show("refreshed", benchmark)
        page = benchmark.items()
        print(f"  members: total={page.total}")
        for item_id in page.item_ids:
            print(f"    - {item_id}")
        if page.total == 0:
            raise RuntimeError(
                "draft has no members; finalize would 400. Did the seed "
                "job skip every item (no ground truth)?"
            )
        assert_has_taxonomy(benchmark)

    flex.step("3. refresh + items (taxonomy + membership)", _read_back)

    # 4. Freeze the draft. Taxonomy (and membership) become immutable.
    def _finalize() -> None:
        show("finalized", benchmark.finalize())
        assert_status(benchmark, "ready")
        assert_has_taxonomy(benchmark)

    flex.step("4. finalize_benchmark", _finalize)

    # 5. A ready benchmark rejects taxonomy changes (server 409).
    flex.expect_raises(
        "5. update rollup after finalize (must fail)",
        NucleusAPIError,
        lambda: benchmark.update(
            rollup_groups=[RollupGroup("Combat Aircraft", ["Attack Aircraft"])]
        ),
    )

    if not args.keep:
        flex.step(
            "6. delete_benchmark (cleanup)",
            lambda: client.delete_benchmark(benchmark.id),
        )
    else:
        print(f"\n--keep: leaving {benchmark.id} in place")

    print("\nDone.")
    print(f"  benchmark: {benchmark.id}")
    print(f"  status:    {benchmark.status}")
    print(f"  taxonomy:  {benchmark.allowed_label_matches_id}")
    raise SystemExit(flex.report())


if __name__ == "__main__":
    main()
