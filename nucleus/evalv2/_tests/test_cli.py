"""Step 10 — run_local_eval + `nu evalv2 run` CLI, isolated from the heavy nucleus SDK.

The evalv2 modules use proper absolute imports (`from nucleus.evalv2.X import ...`). To load them
without executing the heavy ``nucleus/__init__``, we register empty stub packages for ``nucleus`` and
``nucleus.evalv2`` in ``sys.modules``, then load each submodule from its file in dependency order.
"""

import importlib.util
import json
import pathlib
import sys
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

_EVALV2 = pathlib.Path(__file__).resolve().parent.parent


def _stub(name: str) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []  # mark as package
        sys.modules[name] = m


def _load(mod_name: str, filename: str):
    _stub("nucleus")
    _stub("nucleus.evalv2")
    spec = importlib.util.spec_from_file_location(mod_name, _EVALV2 / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # register before exec so downstream imports resolve
    spec.loader.exec_module(mod)
    return mod


_load("nucleus.evalv2.parquet_io", "parquet_io.py")
run_mod = _load("nucleus.evalv2.run", "run.py")
cli_mod = _load("nucleus.evalv2.cli", "cli.py")


def _write_predictions(path, rows):
    cols = {k: [r[k] for r in rows] for k in ["prediction_id", "dataset_item_id", "label", "confidence", "x", "y", "width", "height"]}
    cols["metadata"] = [None] * len(rows)
    pq.write_table(pa.table(cols), path)


def _write_gt(path, rows):
    cols = {k: [r[k] for r in rows] for k in ["ground_truth_id", "dataset_item_id", "label", "x", "y", "width", "height"]}
    cols["metadata"] = [None] * len(rows)
    pq.write_table(pa.table(cols), path)


def _fixture(tmp_path):
    preds = tmp_path / "p.parquet"
    gt = tmp_path / "g.parquet"
    _write_predictions(
        preds,
        [
            {"prediction_id": "p1", "dataset_item_id": "d1", "label": "car", "confidence": 0.9, "x": 0.0, "y": 0.0, "width": 20.0, "height": 20.0},
            {"prediction_id": "p2", "dataset_item_id": "d2", "label": "truck", "confidence": 0.3, "x": 0.0, "y": 0.0, "width": 20.0, "height": 20.0},
        ],
    )
    _write_gt(
        gt,
        [
            {"ground_truth_id": "g1", "dataset_item_id": "d1", "label": "car", "x": 0.0, "y": 0.0, "width": 20.0, "height": 20.0},
            {"ground_truth_id": "g2", "dataset_item_id": "d2", "label": "truck", "x": 0.0, "y": 0.0, "width": 20.0, "height": 20.0},
        ],
    )
    return preds, gt


# ---- config_from_dict -----------------------------------------------------------------------------


def test_config_from_dict_parses_all() -> None:
    cfg = run_mod.config_from_dict(
        {
            "rollup_groups": [{"class_name": "vehicle", "labels": ["car", "truck"]}],
            "exclusion_rules": [{"type": "confidence", "min_confidence": 0.4}],
            "class_agnostic": True,
            "iou_type": "bbox",
            "min_prediction_score": 0.2,
        }
    )
    assert cfg.rollup_groups[0].class_name == "vehicle"
    assert cfg.exclusion_rules and cfg.exclusion_rules[0].type == "confidence"
    assert cfg.class_agnostic is True and cfg.min_prediction_score == 0.2


# ---- run_local_eval -------------------------------------------------------------------------------


def test_run_local_eval_writes_bundle(tmp_path) -> None:
    preds, gt = _fixture(tmp_path)
    out = tmp_path / "out"
    summary = run_mod.run_local_eval(preds, gt, out)
    assert summary.total_gt == 2
    assert (out / "matches.parquet").exists() and (out / "summary.json").exists()


# ---- CLI ------------------------------------------------------------------------------------------


def test_cli_run_smoke(tmp_path) -> None:
    preds, gt = _fixture(tmp_path)
    out = tmp_path / "out"
    res = CliRunner().invoke(cli_mod.evalv2, ["run", "--predictions", str(preds), "--ground-truth", str(gt), "--out", str(out)])
    assert res.exit_code == 0, res.output
    echoed = json.loads(res.output)
    assert echoed["total_gt"] == 2 and echoed["out"] == str(out)
    assert (out / "charts.json").exists()


def test_cli_config_and_flags_honored(tmp_path) -> None:
    preds, gt = _fixture(tmp_path)
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.json"
    # rollup car+truck -> vehicle; without it d1(car) and d2(truck) each match same-label GTs anyway,
    # so exercise min_prediction_score: drop p2 (0.3) via the flag -> d2 GT becomes an FN.
    cfg.write_text(json.dumps({"rollup_groups": [{"class_name": "vehicle", "labels": ["car", "truck"]}]}))
    res = CliRunner().invoke(
        cli_mod.evalv2,
        ["run", "--predictions", str(preds), "--ground-truth", str(gt), "--out", str(out),
         "--config", str(cfg), "--min-prediction-score", "0.5"],
    )
    assert res.exit_code == 0, res.output
    echoed = json.loads(res.output)
    # p1(0.9) kept -> TP; p2(0.3) dropped -> d2 GT is FN
    assert echoed["total_tp"] == 1 and echoed["total_fn"] == 1


def test_cli_missing_file_errors() -> None:
    res = CliRunner().invoke(cli_mod.evalv2, ["run", "--predictions", "/nope.parquet", "--ground-truth", "/nope2.parquet", "--out", "/tmp/x"])
    assert res.exit_code != 0
