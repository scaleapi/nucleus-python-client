"""Programmatic entry for a local EvaluationV2 run: parquet in -> result bundle out.

Wires the parquet adapter to the shared ``evalv2_core`` kernel. Import-light (``evalv2_core`` +
``parquet_io``); does not import the heavy top-level ``nucleus`` SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalv2_core import run as core_run
from evalv2_core.exclusions import parse_exclusion_rules
from evalv2_core.types import EvalConfig, EvalSummary, RollupGroup

from nucleus.evalv2.parquet_io import ParquetSink, ParquetSource


def config_from_dict(raw: dict[str, Any] | None) -> EvalConfig:
    """Build an EvalConfig from a plain dict (CLI --config JSON). All keys optional."""
    d = raw or {}
    kwargs: dict[str, Any] = {}
    if d.get("rollup_groups"):
        kwargs["rollup_groups"] = [
            RollupGroup(class_name=g["class_name"], labels=list(g["labels"]))
            for g in d["rollup_groups"]
        ]
    if d.get("exclusion_rules"):
        kwargs["exclusion_rules"] = parse_exclusion_rules(d["exclusion_rules"])
    if "class_agnostic" in d:
        kwargs["class_agnostic"] = bool(d["class_agnostic"])
    if d.get("iou_type"):
        kwargs["iou_type"] = d["iou_type"]
    if d.get("min_prediction_score") is not None:
        kwargs["min_prediction_score"] = float(d["min_prediction_score"])
    if d.get("confidence_grid"):
        kwargs["confidence_grid"] = tuple(float(x) for x in d["confidence_grid"])
    return EvalConfig(**kwargs)


def run_local_eval(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    out_dir: str | Path,
    *,
    items_path: str | Path | None = None,
    config: EvalConfig | None = None,
    shard_size: int = 2000,
) -> EvalSummary:
    """Run an evaluation over local (sorted) parquets and write the result bundle to ``out_dir``."""
    source = ParquetSource(
        predictions_path, ground_truth_path, items_path=items_path, config=config or EvalConfig()
    )
    sink = ParquetSink(out_dir)
    return core_run(source, sink, shard_size=shard_size)
