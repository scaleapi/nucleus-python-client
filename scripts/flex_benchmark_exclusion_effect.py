#!/usr/bin/env python
"""Prove benchmark exclusion rules actually drop content from the *benchmark's own views*.

Companion to flex_benchmark_exclusions.py (which flexes the create/update/validation API surface).
This one is a live, eyeball-it verification of the piece-2/3 backend change that makes a benchmark's
Insights / ground-truth coverage / item grid honor its own exclusion rules — not just evaluations.

It finalizes TWO benchmarks over the SAME member items:

  * control  — no exclusion rules
  * excluded — one annotation-scope groundTruth label rule dropping "Bomber Aircraft"

Both are created as `ready` (draft=False), so the server runs the build job and recomputes the frozen
content metrics through the new exclusion filter. Then it prints each benchmark id and its Insights
URL so you can compare the label distribution:

  control  : 5 classes / 27 annotations, Bomber Aircraft 8 (30%)
  excluded : 4 classes / 19 annotations, no Bomber Aircraft bucket

(Exact counts depend on the seed GT; the point is control minus excluded == the Bomber Aircraft count.)

IMPORTANT: the backend must be running the piece-2/3 code. Because exclusions are applied at metric
recompute (finalize/build), a benchmark finalized by an OLD backend will still show the full 27 — so
restart the backend before running this.

Usage:
    # uses $NUCLEUS_PRODCLONE_API_KEY + local prodclone backend by default
    python scripts/flex_benchmark_exclusion_effect.py

    # clean them up afterwards instead of leaving them for inspection
    python scripts/flex_benchmark_exclusion_effect.py --delete
"""
import argparse
import os
import time

from nucleus import LabelExclusionRule, NucleusClient

# Same scattered prodclone items the sibling flex scripts use (ds_d6ng124awzzg0gtc8x50, all with GT).
ITEM_IDS = [
    "di_d6ng124z6a1006gm86ag",
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
    "di_d6ng124z6a1006gm86c0",
    "di_d6ng124z6a1006gm86e0",
    "di_d6ng124z6a1006gm86eg",
]

# The class we drop. Annotation-scope groundTruth => individual "Bomber Aircraft" GT boxes disappear
# from the benchmark's content views (the item stays a member via its other-class GT).
EXCLUDED_LABEL = "Bomber Aircraft"
BOMBER_RULE = LabelExclusionRule(
    scope="annotation", target="groundTruth", labels=[EXCLUDED_LABEL]
)


def make_ready_benchmark(client, name, description, exclusion_rules):
    """Create a finalized (ready) benchmark and block until its build/recompute completes."""
    print(f"\nCreating {name!r} (ready)...")
    bm = client.create_benchmark(
        name,
        description=description,
        metadata={"purpose": "flex_benchmark_exclusion_effect.py"},
        item_ids=ITEM_IDS,
        exclusion_rules=exclusion_rules,  # None => omit; a list => set
        # draft defaults to False -> server builds + recomputes content metrics now.
    )
    print(
        f"  {bm.id} status={bm.status!r} items={bm.item_count} "
        f"exclusion_rules={len(bm.exclusion_rules or [])}"
    )
    return bm


def insights_url(base_ui: str, spoof: str, bm_id: str) -> str:
    return f"{base_ui}/pubsec-data-engine/{bm_id}?spoof={spoof}&bm_tab=insights"


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
        "--ui-base",
        default="http://localhost:3003",
        help="Base URL of the pubsec UI, for the printed Insights links.",
    )
    parser.add_argument(
        "--spoof",
        default="66ab902a5d92f6f39ca2ef9a",
        help="Spoof user id for the printed Insights links.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete both benchmarks at the end instead of leaving them for inspection.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "no API key: pass --api-key or set NUCLEUS_PRODCLONE_API_KEY"
        )

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)
    stamp = int(time.time())

    control = make_ready_benchmark(
        client,
        f"flex-excl-effect-{stamp}-control",
        "control: no exclusions (expect full GT)",
        exclusion_rules=None,
    )
    excluded = make_ready_benchmark(
        client,
        f"flex-excl-effect-{stamp}-excluded",
        f"excluded: drop {EXCLUDED_LABEL} (expect fewer GT)",
        exclusion_rules=[BOMBER_RULE],
    )

    print("\n" + "=" * 68)
    print("Open both Insights tabs and compare the label distribution:")
    print(f"\n  control  ({control.id}) — expect all classes incl. {EXCLUDED_LABEL}:")
    print(f"    {insights_url(args.ui_base, args.spoof, control.id)}")
    print(f"\n  excluded ({excluded.id}) — expect NO {EXCLUDED_LABEL}, fewer annotations:")
    print(f"    {insights_url(args.ui_base, args.spoof, excluded.id)}")
    print(
        "\nThe difference in total annotations between the two == the number of "
        f"{EXCLUDED_LABEL!r}\nGT boxes across these items. If the excluded benchmark still shows "
        f"{EXCLUDED_LABEL},\nthe backend that finalized it predates the piece-2/3 change — restart it "
        "and rerun."
    )

    if args.delete:
        print("\n--delete: cleaning up...")
        for bm in (control, excluded):
            client.delete_benchmark(bm.id)
            print(f"  deleted {bm.id}")
    else:
        print(f"\nLeaving both benchmarks in place (pass --delete to remove).")


if __name__ == "__main__":
    main()
