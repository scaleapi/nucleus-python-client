"""Pydantic models for Nucleus Evaluations V2 REST payloads."""

from typing import Any, Dict, List, Literal, Optional

from nucleus.pydantic_base import DictCompatibleModel


class RangeNum(DictCompatibleModel):
    min: Optional[float] = None
    max: Optional[float] = None


class MetadataPredicate(DictCompatibleModel):
    key: str
    op: Literal["EQ", "IN", "GT", "LT"]
    value: Optional[Any] = None


class EvaluationV2FilterArgs(DictCompatibleModel):
    """Filter object for charts/examples calls (mirrors server evaluation_v2 SQL filters)."""

    confidence_range: Optional[RangeNum] = None
    iou_range: Optional[RangeNum] = None
    pred_labels: Optional[List[str]] = None
    gt_labels: Optional[List[str]] = None
    item_metadata: Optional[List[MetadataPredicate]] = None
    prediction_metadata: Optional[List[MetadataPredicate]] = None
    label_equality: Optional[Literal["EQ", "NEQ"]] = None
    has_ground_truth: Optional[bool] = None
    tide_background: Optional[bool] = None

    def to_api_filters(self) -> Dict[str, Any]:
        """Serialize to camelCase keys expected by the GraphQL / REST layer."""
        d = self.dict(exclude_none=True)
        # pydantic v1 uses snake_case fields; server expects camelCase in JSON filters
        out: Dict[str, Any] = {}
        if "confidence_range" in d:
            out["confidenceRange"] = d["confidence_range"]
        if "iou_range" in d:
            out["iouRange"] = d["iou_range"]
        if "pred_labels" in d:
            out["predLabels"] = d["pred_labels"]
        if "gt_labels" in d:
            out["gtLabels"] = d["gt_labels"]
        if "item_metadata" in d:
            out["itemMetadata"] = d["item_metadata"]
        if "prediction_metadata" in d:
            out["predictionMetadata"] = d["prediction_metadata"]
        if "label_equality" in d:
            out["labelEquality"] = d["label_equality"]
        if "has_ground_truth" in d:
            out["hasGroundTruth"] = d["has_ground_truth"]
        if "tide_background" in d:
            out["tideBackground"] = d["tide_background"]
        return out


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
    item_metadata: Dict[str, Any]
    prediction_metadata: Dict[str, Any]
    prediction_row: Optional[Dict[str, Any]] = None
    annotation_row: Optional[Dict[str, Any]] = None


class EvaluationV2ExamplesPage(DictCompatibleModel):
    rows: List[EvaluationV2MatchExample]
    total: int
