"""Response and filter models for Evaluation V2."""

from typing import Any, Dict, List, Literal, Optional

from nucleus.pydantic_base import DictCompatibleModel


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    if len(parts) == 1:
        return name
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _camelize_filter_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _snake_to_camel(key): (
                val if key == "value" else _camelize_filter_value(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_camelize_filter_value(item) for item in value]
    return value


class RangeNum(DictCompatibleModel):
    min: Optional[float] = None
    max: Optional[float] = None


class MetadataPredicate(DictCompatibleModel):
    key: str
    op: Literal["EQ", "IN", "GT", "LT"]
    value: Optional[Any] = None


_FILTER_API_KEYS = {
    "confidence_range": "confidenceRange",
    "iou_range": "iouRange",
    "pred_labels": "predLabels",
    "gt_labels": "gtLabels",
    "item_metadata": "itemMetadata",
    "prediction_metadata": "predictionMetadata",
    "gt_area_range": "gtAreaRange",
    "label_equality": "labelEquality",
    "has_ground_truth": "hasGroundTruth",
    "tide_background": "tideBackground",
    "slice_ids": "sliceIds",
}


class EvaluationV2FilterArgs(DictCompatibleModel):
    """Optional filters for :meth:`nucleus.evaluation_v2.EvaluationV2.charts` and :meth:`nucleus.evaluation_v2.EvaluationV2.examples`."""

    confidence_range: Optional[RangeNum] = None
    iou_range: Optional[RangeNum] = None
    pred_labels: Optional[List[str]] = None
    gt_labels: Optional[List[str]] = None
    item_metadata: Optional[List[MetadataPredicate]] = None
    prediction_metadata: Optional[List[MetadataPredicate]] = None
    gt_area_range: Optional[RangeNum] = None
    label_equality: Optional[Literal["EQ", "NEQ"]] = None
    has_ground_truth: Optional[bool] = None
    tide_background: Optional[bool] = None
    slice_ids: Optional[List[str]] = None

    def to_api_filters(self) -> Dict[str, Any]:
        """Return filters as a dict ready for API requests."""
        d = self.dict(exclude_none=True)
        return {
            api_key: _camelize_filter_value(d[snake_key])
            for snake_key, api_key in _FILTER_API_KEYS.items()
            if snake_key in d
        }


class MapSummary(DictCompatibleModel):
    mapAt50: Optional[float] = None
    mapAt75: Optional[float] = None
    mapAt5095: Optional[float] = None


class PerClassAp(DictCompatibleModel):
    classLabel: str
    ap: float


class ConfusionEntry(DictCompatibleModel):
    gtLabel: str
    predLabel: str
    count: int


class ScoreHistogramBucket(DictCompatibleModel):
    bucketMin: float
    bucketMax: float
    count: int


class TotalCounts(DictCompatibleModel):
    tp: int
    fp: int
    fn: int
    predsWithConfidence: int


class ApBySize(DictCompatibleModel):
    small: Optional[float] = None
    medium: Optional[float] = None
    large: Optional[float] = None


class PrCurvePoint(DictCompatibleModel):
    classLabel: str
    recall: float
    precision: float


class TideAttribution(DictCompatibleModel):
    truePositive: int
    localization: int
    classification: int
    both: int
    duplicate: int
    background: int
    missed: int


class EvaluationV2Charts(DictCompatibleModel):
    mapSummary: MapSummary
    perClassAp: List[PerClassAp]
    confusionMatrix: List[ConfusionEntry]
    scoreHistogram: List[ScoreHistogramBucket]
    computedIouRanges: List[float]
    totalCounts: TotalCounts
    apBySize: ApBySize
    prCurve: List[PrCurvePoint]
    tideAttribution: TideAttribution


class EvaluationV2MatchExample(DictCompatibleModel):
    id: str
    evaluation_id: str
    dataset_item_id: str
    model_prediction_id: Optional[str] = None
    ground_truth_annotation_id: Optional[str] = None
    pred_canonical_label: Optional[str] = None
    gt_canonical_label: Optional[str] = None
    pred_raw_label: Optional[str] = None
    gt_raw_label: Optional[str] = None
    iou: Optional[float] = None
    confidence: Optional[float] = None
    true_positive: bool
    match_type: str
    gt_area: Optional[float] = None
    item_metadata: Optional[Dict[str, Any]] = None
    prediction_metadata: Optional[Dict[str, Any]] = None
    prediction_row: Optional[Dict[str, Any]] = None
    annotation_row: Optional[Dict[str, Any]] = None


class EvaluationV2ExamplesPage(DictCompatibleModel):
    rows: List[EvaluationV2MatchExample]
    total: int
