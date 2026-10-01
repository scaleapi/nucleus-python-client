"""Parquet adapter for the local (SDK) EvaluationV2 harness.

Implements the evalv2-core ``EvalSource``/``EvalSink`` ports over parquet:

  - ``ParquetSource`` reads a predictions parquet + a ground-truth parquet (+ optional items parquet),
    joins them on ``dataset_item_id``, and yields ``ItemBundle``s. Config is a constructor argument
    (never stored in the parquet).
  - ``ParquetSink`` streams match rows / per-threshold rows into parquet files and writes the
    summary + charts as JSON — the result bundle the "upload eval" flow later pushes to the platform.

Depends only on ``evalv2_core`` + ``pyarrow`` — never the heavy top-level ``nucleus`` SDK.

Scale: ``ParquetSource`` streams via a sorted merge-join — it never loads a whole parquet into memory.
The input parquets MUST be sorted by ``dataset_item_id`` (string order); the source iterates all files
in row-group batches in lock-step, buffering only the current item's rows (memory is O(one item +
batch), independent of dataset size). Out-of-order input raises loudly rather than silently
mis-joining.

Input schema (columns; each file sorted by dataset_item_id):
  predictions.parquet : prediction_id, dataset_item_id, label, confidence, x, y, width, height,
                        [segmentation (JSON), metadata (JSON)]
  ground_truth.parquet: ground_truth_id, dataset_item_id, label, x, y, width, height,
                        [segmentation (JSON), metadata (JSON)]
  items.parquet (opt) : dataset_item_id, metadata (JSON)

Output bundle (out_dir):
  matches.parquet, per_threshold.parquet, summary.json, charts.json
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from evalv2_core.types import (
    BoxGeometry,
    ChartsPayload,
    EvalConfig,
    EvalSummary,
    Geometry,
    GroundTruth,
    ItemBundle,
    MatchRow,
    MetadataField,
    MetricAtConfidence,
    PerThresholdRow,
    Prediction,
    SegmentationGeometry,
)

# ---- parquet output schemas ----------------------------------------------------------------------

_MATCH_SCHEMA = pa.schema(
    [
        ("dataset_item_id", pa.string()),
        ("match_type", pa.string()),
        ("iou", pa.float64()),
        ("true_positive", pa.bool_()),
        ("model_prediction_id", pa.string()),
        ("ground_truth_annotation_id", pa.string()),
        ("pred_canonical_label", pa.string()),
        ("gt_canonical_label", pa.string()),
        ("pred_raw_label", pa.string()),
        ("gt_raw_label", pa.string()),
        ("confidence", pa.float64()),
        ("gt_area", pa.float64()),
        ("pred_area", pa.float64()),
        ("item_metadata", pa.string()),  # JSON
        ("prediction_metadata", pa.string()),  # JSON
        ("nearest_same_class_gt_id", pa.string()),
        ("nearest_same_class_gt_iou", pa.float64()),
        ("nearest_any_class_gt_id", pa.string()),
        ("nearest_any_class_gt_iou", pa.float64()),
        ("nearest_any_class_gt_canonical_label", pa.string()),
    ]
)

_PER_THRESHOLD_SCHEMA = pa.schema(
    [
        ("iou_threshold", pa.float64()),
        ("dataset_item_id", pa.string()),
        ("model_prediction_id", pa.string()),
        ("pred_canonical_label", pa.string()),
        ("confidence", pa.float64()),
        ("matched_gt_id", pa.string()),
        ("matched_gt_canonical_label", pa.string()),
        ("iou", pa.float64()),
        ("size_bucket", pa.string()),
    ]
)


# ---- helpers -------------------------------------------------------------------------------------


def _json_load(value: Any) -> dict:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(value)


def _geometry_from_row(row: dict) -> Geometry:
    seg = row.get("segmentation")
    if seg not in (None, ""):
        return SegmentationGeometry(segmentation=seg if isinstance(seg, (list, dict)) else json.loads(seg))
    return BoxGeometry(
        x=float(row["x"]), y=float(row["y"]), width=float(row["width"]), height=float(row["height"])
    )


def _iter_item_groups(path: str, source_name: str, batch_size: int) -> Iterator[tuple[str, list[dict]]]:
    """Yield (dataset_item_id, rows) for each contiguous block, streaming row-group batches.

    Requires the file sorted by ``dataset_item_id`` (string order). Buffers only the current item's
    rows; raises if a group boundary goes backwards (unsorted input) to avoid silent mis-joins.
    """
    current_id: str | None = None
    buf: list[dict] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            rid = row["dataset_item_id"]
            if current_id is None:
                current_id, buf = rid, [row]
            elif rid == current_id:
                buf.append(row)
            else:
                if rid < current_id:
                    raise ValueError(
                        f"{source_name} must be sorted by dataset_item_id "
                        f"(saw {rid!r} after {current_id!r})"
                    )
                yield current_id, buf
                current_id, buf = rid, [row]
    if current_id is not None:
        yield current_id, buf


class _PeekableGroups:
    """One-item lookahead over an ``_iter_item_groups`` generator."""

    def __init__(self, gen: Iterator[tuple[str, list[dict]]]) -> None:
        self._gen = gen
        self._current: tuple[str, list[dict]] | None = None
        self._advance()

    def _advance(self) -> None:
        self._current = next(self._gen, None)

    @property
    def current_id(self) -> str | None:
        return self._current[0] if self._current is not None else None

    def take(self) -> list[dict]:
        assert self._current is not None
        rows = self._current[1]
        self._advance()
        return rows


# ---- source --------------------------------------------------------------------------------------


class ParquetSource:
    """EvalSource over two (or three) parquet files. Config is supplied here, not read from parquet."""

    def __init__(
        self,
        predictions_path: str | Path,
        ground_truth_path: str | Path,
        items_path: str | Path | None = None,
        config: EvalConfig | None = None,
        *,
        batch_size: int = 65_536,
    ) -> None:
        self._predictions_path = str(predictions_path)
        self._ground_truth_path = str(ground_truth_path)
        self._items_path = str(items_path) if items_path is not None else None
        self._config = config or EvalConfig()
        self._batch_size = batch_size

    def read_config(self) -> EvalConfig:
        return self._config

    def read_items(self) -> Iterator[ItemBundle]:
        """Streaming sorted merge-join over the (sorted) parquets. Memory is O(one item + batch)."""
        preds = _PeekableGroups(
            _iter_item_groups(self._predictions_path, "predictions.parquet", self._batch_size)
        )
        gts = _PeekableGroups(
            _iter_item_groups(self._ground_truth_path, "ground_truth.parquet", self._batch_size)
        )
        items = _PeekableGroups(
            _iter_item_groups(self._items_path, "items.parquet", self._batch_size)
            if self._items_path is not None
            else iter(())
        )

        while True:
            ids = [g.current_id for g in (preds, gts, items) if g.current_id is not None]
            if not ids:
                return
            di = min(ids)  # lexicographic — matches the sorted-string contract
            pred_rows = preds.take() if preds.current_id == di else []
            gt_rows = gts.take() if gts.current_id == di else []
            meta_rows = items.take() if items.current_id == di else []
            yield ItemBundle(
                dataset_item_id=di,
                predictions=[
                    Prediction(
                        id=r["prediction_id"],
                        dataset_item_id=di,
                        label=r["label"],
                        confidence=float(r["confidence"]),
                        geometry=_geometry_from_row(r),
                        metadata=_json_load(r.get("metadata")),
                    )
                    for r in pred_rows
                ],
                ground_truths=[
                    GroundTruth(
                        id=r["ground_truth_id"],
                        dataset_item_id=di,
                        label=r["label"],
                        geometry=_geometry_from_row(r),
                        metadata=_json_load(r.get("metadata")),
                    )
                    for r in gt_rows
                ],
                item_metadata=_json_load(meta_rows[0].get("metadata")) if meta_rows else {},
            )


# ---- sink ----------------------------------------------------------------------------------------


def _match_row_to_dict(r: MatchRow) -> dict:
    d = asdict(r)
    d["item_metadata"] = json.dumps(d["item_metadata"])
    d["prediction_metadata"] = json.dumps(d["prediction_metadata"])
    return d


class ParquetSink:
    """EvalSink writing the result bundle (matches/per_threshold parquet + summary/charts JSON)."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self._match_writer: pq.ParquetWriter | None = None
        self._per_threshold_writer: pq.ParquetWriter | None = None

    def begin(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._match_writer = pq.ParquetWriter(self.out_dir / "matches.parquet", _MATCH_SCHEMA)
        self._per_threshold_writer = pq.ParquetWriter(
            self.out_dir / "per_threshold.parquet", _PER_THRESHOLD_SCHEMA
        )

    def write_matches(
        self,
        match_rows: Iterable[MatchRow],
        per_threshold_rows: Iterable[PerThresholdRow],
    ) -> None:
        assert self._match_writer is not None and self._per_threshold_writer is not None, "call begin()"
        match_dicts = [_match_row_to_dict(r) for r in match_rows]
        if match_dicts:
            self._match_writer.write_table(pa.Table.from_pylist(match_dicts, schema=_MATCH_SCHEMA))
        per_thr_dicts = [asdict(r) for r in per_threshold_rows]
        if per_thr_dicts:
            self._per_threshold_writer.write_table(
                pa.Table.from_pylist(per_thr_dicts, schema=_PER_THRESHOLD_SCHEMA)
            )

    def write_summary(
        self,
        summary: EvalSummary,
        metrics_at_confidence: Sequence[MetricAtConfidence],
        metadata_fields: Sequence[MetadataField],
        charts: ChartsPayload,
    ) -> None:
        summary_doc = {
            "summary": asdict(summary),
            "metrics_at_confidence": [asdict(m) for m in metrics_at_confidence],
            "metadata_fields": [asdict(f) for f in metadata_fields],
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary_doc, indent=2))
        (self.out_dir / "charts.json").write_text(json.dumps(dict(charts.payload), indent=2))

    def finalize(self) -> None:
        if self._match_writer is not None:
            self._match_writer.close()
            self._match_writer = None
        if self._per_threshold_writer is not None:
            self._per_threshold_writer.close()
            self._per_threshold_writer = None
