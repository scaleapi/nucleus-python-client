"""Exclusion rules for Evaluation V2 creation.

These rules drop items/annotations from an evaluation before metrics are computed.

The per-rule shape is validated server-side at create time
(``parseEvaluationV2ExclusionRulesWithDiagnostics``), which reports exactly which
rules were rejected and why — so these classes only need to serialize correctly.

Pass instances (or equivalent plain dicts) to
:meth:`nucleus.NucleusClient.create_evaluation_v2` via ``exclusion_rules``::

    client.create_evaluation_v2(
        model_run_id,
        exclusion_rules=[
            BoxAreaExclusionRule(scope="annotation", target="groundTruth", min=1024),
            LabelExclusionRule(scope="item", target="prediction", labels=["ignore"]),
            MetadataExclusionRule(key="is_dark", op="EQ", value=True),
        ],
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# String literals are sent as values (not keys), so the server's request-body
# camelcaser preserves them verbatim — emit them exactly as the backend expects.
ExclusionScope = str  # "item" | "annotation"
ExclusionTarget = str  # "groundTruth" | "prediction"
MetadataOp = str  # "EQ" | "IN" | "GT" | "LT"


@dataclass
class MetadataExclusionRule:
    """Exclude whole items whose item-metadata ``key`` matches ``value`` under ``op``.

    ``scope`` is always ``"item"`` for metadata rules.
    """

    key: str
    op: MetadataOp
    value: Any
    scope: ExclusionScope = "item"

    def to_api_dict(self) -> Dict[str, Any]:
        return {
            "type": "metadata",
            "scope": self.scope,
            "key": self.key,
            "op": self.op,
            "value": self.value,
        }


@dataclass
class LabelExclusionRule:
    """Exclude annotations/predictions (or whole items) carrying any of ``labels``.

    Parameters:
        scope: ``"item"`` (drop the whole item if any annotation matches) or
            ``"annotation"`` (drop only matching annotations).
        target: ``"groundTruth"`` or ``"prediction"`` — which side to filter.
        labels: Labels to exclude.
    """

    scope: ExclusionScope
    target: ExclusionTarget
    labels: List[str] = field(default_factory=list)

    def to_api_dict(self) -> Dict[str, Any]:
        return {
            "type": "labels",
            "scope": self.scope,
            "target": self.target,
            "labels": list(self.labels),
        }


@dataclass
class BoxAreaExclusionRule:
    """Exclude boxes whose pixel area falls outside ``[min, max]`` (at least one bound required).

    Parameters:
        scope: ``"item"`` or ``"annotation"``.
        target: ``"groundTruth"`` or ``"prediction"``.
        min: Minimum pixel area (inclusive lower bound), or ``None``.
        max: Maximum pixel area (inclusive upper bound), or ``None``.
    """

    scope: ExclusionScope
    target: ExclusionTarget
    min: Optional[float] = None
    max: Optional[float] = None

    def to_api_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": "boxArea",
            "scope": self.scope,
            "target": self.target,
        }
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        return out


EvaluationV2ExclusionRule = Union[
    MetadataExclusionRule,
    LabelExclusionRule,
    BoxAreaExclusionRule,
]
