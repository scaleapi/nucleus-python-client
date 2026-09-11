"""`nu evalv2` CLI: run a local EvaluationV2 over parquet.

Registered into the main ``nu`` CLI (cli/nu.py). Import-light; the heavy ``nucleus`` SDK is not
imported here.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from nucleus.evalv2.run import config_from_dict, run_local_eval


@click.group("evalv2")
def evalv2() -> None:
    """Local EvaluationV2 over exported parquet (offline; upload separately)."""


@evalv2.command("run")
@click.option("--predictions", required=True, type=click.Path(exists=True), help="predictions.parquet")
@click.option("--ground-truth", required=True, type=click.Path(exists=True), help="ground_truth.parquet")
@click.option("--items", type=click.Path(exists=True), default=None, help="optional items.parquet (metadata)")
@click.option("--out", required=True, type=click.Path(), help="output directory for the result bundle")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None, help="JSON EvalConfig")
@click.option("--iou-type", type=click.Choice(["bbox", "segm"]), default=None)
@click.option("--class-agnostic", is_flag=True, default=False)
@click.option("--min-prediction-score", type=float, default=None)
@click.option("--shard-size", type=int, default=2000, show_default=True)
def run_cmd(
    predictions: str,
    ground_truth: str,
    items: str | None,
    out: str,
    config_path: str | None,
    iou_type: str | None,
    class_agnostic: bool,
    min_prediction_score: float | None,
    shard_size: int,
) -> None:
    """Compute an evaluation locally and write matches/summary/charts to OUT."""
    cfg_dict = json.loads(Path(config_path).read_text()) if config_path else {}
    if iou_type:
        cfg_dict["iou_type"] = iou_type
    if class_agnostic:
        cfg_dict["class_agnostic"] = True
    if min_prediction_score is not None:
        cfg_dict["min_prediction_score"] = min_prediction_score

    summary = run_local_eval(
        predictions,
        ground_truth,
        out,
        items_path=items,
        config=config_from_dict(cfg_dict),
        shard_size=shard_size,
    )
    click.echo(
        json.dumps(
            {
                "map_50": summary.map_50,
                "map_50_95": summary.map_50_95,
                "total_tp": summary.total_tp,
                "total_fp": summary.total_fp,
                "total_fn": summary.total_fn,
                "total_gt": summary.total_gt,
                "out": str(out),
            },
            indent=2,
        )
    )
