"""Round-trip test for the parquet adapter.

Isolated from the heavy top-level ``nucleus`` package: parquet_io is loaded directly from its file
(so importing it never triggers ``nucleus/__init__``), and pytest is run with ``--confcutdir`` so the
repo-root ``conftest.py`` (which imports nucleus + requires an API key) is never collected.
"""

import importlib.util
import json
import pathlib
from collections.abc import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import evalv2_core
from evalv2_core.types import (
    BoxGeometry,
    EvalConfig,
    EvalSummary,
    GroundTruth,
    ItemBundle,
    Prediction,
)

# --- load parquet_io.py directly, bypassing nucleus/__init__ --------------------------------------
_PARQUET_IO = pathlib.Path(__file__).resolve().parent.parent / "parquet_io.py"
_spec = importlib.util.spec_from_file_location("evalv2_parquet_io", _PARQUET_IO)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ParquetSource = _mod.ParquetSource
ParquetSink = _mod.ParquetSink


# --- in-memory reference source/sink (for round-trip comparison) -----------------------------------
class _MemSource:
    def __init__(self, cfg, items):
        self._cfg, self._items = cfg, items

    def read_config(self):
        return self._cfg

    def read_items(self):
        return iter(self._items)


class _MemSink:
    def __init__(self):
        self.summary = None

    def begin(self):
        ...

    def write_matches(self, m, p):
        ...

    def write_summary(self, summary, metrics, fields, charts):
        self.summary = summary

    def finalize(self):
        ...


def _write_predictions(path):
    pq.write_table(
        pa.table(
            {
                "prediction_id": ["p1", "p2"],
                "dataset_item_id": ["d1", "d2"],
                "label": ["car", "car"],
                "confidence": [0.9, 0.8],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "width": [20.0, 20.0],
                "height": [20.0, 20.0],
                "metadata": [None, None],
            }
        ),
        path,
    )


def _write_ground_truth(path):
    pq.write_table(
        pa.table(
            {
                "ground_truth_id": ["g1", "g2", "g3"],
                "dataset_item_id": ["d1", "d2", "d3"],
                "label": ["car", "person", "car"],
                "x": [0.0, 0.0, 0.0],
                "y": [0.0, 0.0, 0.0],
                "width": [20.0, 20.0, 20.0],
                "height": [20.0, 20.0, 20.0],
                "metadata": [None, None, None],
            }
        ),
        path,
    )


def _write_items(path):
    pq.write_table(
        pa.table(
            {
                "dataset_item_id": ["d1", "d2", "d3"],
                "metadata": [json.dumps({"weather": "rain"}), json.dumps({"weather": "sun"}), json.dumps({"weather": "sun"})],
            }
        ),
        path,
    )


def _reference_items():
    def box(*b):
        return BoxGeometry(*b)

    return [
        ItemBundle("d1", [Prediction("p1", "d1", "car", 0.9, box(0, 0, 20, 20))], [GroundTruth("g1", "d1", "car", box(0, 0, 20, 20))], {"weather": "rain"}),
        ItemBundle("d2", [Prediction("p2", "d2", "car", 0.8, box(0, 0, 20, 20))], [GroundTruth("g2", "d2", "person", box(0, 0, 20, 20))], {"weather": "sun"}),
        ItemBundle("d3", [], [GroundTruth("g3", "d3", "car", box(0, 0, 20, 20))], {"weather": "sun"}),
    ]


def test_parquet_round_trip_matches_in_memory(tmp_path) -> None:
    pred_p = tmp_path / "predictions.parquet"
    gt_p = tmp_path / "ground_truth.parquet"
    items_p = tmp_path / "items.parquet"
    out = tmp_path / "out"
    _write_predictions(pred_p)
    _write_ground_truth(gt_p)
    _write_items(items_p)

    # run through the parquet adapter
    src = ParquetSource(pred_p, gt_p, items_p, config=EvalConfig())
    sink = ParquetSink(out)
    parquet_summary = evalv2_core.run(src, sink)

    # run the same items purely in memory
    mem_sink = _MemSink()
    evalv2_core.run(_MemSource(EvalConfig(), _reference_items()), mem_sink)
    mem_summary = mem_sink.summary

    assert isinstance(parquet_summary, EvalSummary)
    for key in ("map_50", "map_50_95", "total_tp", "total_fp", "total_fn", "total_gt"):
        a, b = getattr(parquet_summary, key), getattr(mem_summary, key)
        assert (a is None and b is None) or a == pytest.approx(b), key


def test_source_yields_joined_bundles(tmp_path) -> None:
    _write_predictions(tmp_path / "p.parquet")
    _write_ground_truth(tmp_path / "g.parquet")
    _write_items(tmp_path / "i.parquet")
    src = ParquetSource(tmp_path / "p.parquet", tmp_path / "g.parquet", tmp_path / "i.parquet")
    items = list(src.read_items())
    assert [it.dataset_item_id for it in items] == ["d1", "d2", "d3"]
    d1 = items[0]
    assert d1.predictions[0].id == "p1" and d1.ground_truths[0].id == "g1"
    assert d1.item_metadata == {"weather": "rain"}
    assert isinstance(d1.predictions[0].geometry, BoxGeometry)


def test_output_bundle_files_written_and_readable(tmp_path) -> None:
    _write_predictions(tmp_path / "p.parquet")
    _write_ground_truth(tmp_path / "g.parquet")
    out = tmp_path / "out"
    src = ParquetSource(tmp_path / "p.parquet", tmp_path / "g.parquet")
    evalv2_core.run(src, ParquetSink(out))

    assert (out / "matches.parquet").exists()
    assert (out / "per_threshold.parquet").exists()
    matches = pq.read_table(out / "matches.parquet").to_pylist()
    assert matches and {m["match_type"] for m in matches} <= {"TP", "FP", "FN"}
    per_thr = pq.read_table(out / "per_threshold.parquet")
    assert "size_bucket" in per_thr.column_names

    summary_doc = json.loads((out / "summary.json").read_text())
    assert "summary" in summary_doc and len(summary_doc["metrics_at_confidence"]) == 17
    charts = json.loads((out / "charts.json").read_text())
    for key in ("mapSummary", "prCurve", "confusionMatrix", "tideAttribution", "f1Curve"):
        assert key in charts


def test_streaming_groups_across_batch_boundaries(tmp_path) -> None:
    # force many tiny batches so an item's rows and item boundaries span batches
    _write_predictions(tmp_path / "p.parquet")
    _write_ground_truth(tmp_path / "g.parquet")
    _write_items(tmp_path / "i.parquet")
    src = ParquetSource(
        tmp_path / "p.parquet", tmp_path / "g.parquet", tmp_path / "i.parquet", batch_size=1
    )
    items = list(src.read_items())
    assert [it.dataset_item_id for it in items] == ["d1", "d2", "d3"]
    assert items[0].predictions[0].id == "p1" and items[0].ground_truths[0].id == "g1"
    assert items[2].predictions == [] and items[2].ground_truths[0].id == "g3"  # d3 gt-only


def test_unsorted_input_raises(tmp_path) -> None:
    # predictions out of dataset_item_id order -> loud error, not a silent mis-join
    pq.write_table(
        pa.table(
            {
                "prediction_id": ["p2", "p1"],
                "dataset_item_id": ["d2", "d1"],  # descending -> unsorted
                "label": ["car", "car"],
                "confidence": [0.8, 0.9],
                "x": [0.0, 0.0], "y": [0.0, 0.0], "width": [20.0, 20.0], "height": [20.0, 20.0],
                "metadata": [None, None],
            }
        ),
        tmp_path / "p.parquet",
    )
    _write_ground_truth(tmp_path / "g.parquet")
    src = ParquetSource(tmp_path / "p.parquet", tmp_path / "g.parquet", batch_size=1)
    with pytest.raises(ValueError, match="sorted by dataset_item_id"):
        list(src.read_items())


def test_item_metadata_round_trips_into_match_rows(tmp_path) -> None:
    _write_predictions(tmp_path / "p.parquet")
    _write_ground_truth(tmp_path / "g.parquet")
    _write_items(tmp_path / "i.parquet")
    out = tmp_path / "out"
    src = ParquetSource(tmp_path / "p.parquet", tmp_path / "g.parquet", tmp_path / "i.parquet")
    evalv2_core.run(src, ParquetSink(out))
    matches = pq.read_table(out / "matches.parquet").to_pylist()
    d1_rows = [m for m in matches if m["dataset_item_id"] == "d1"]
    assert d1_rows and json.loads(d1_rows[0]["item_metadata"]) == {"weather": "rain"}
