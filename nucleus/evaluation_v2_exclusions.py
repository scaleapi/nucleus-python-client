"""Exclusion rules for Evaluation V2 creation.

These rules drop items/annotations from an evaluation before metrics are computed.

Each rule is validated when the evaluation is created; anything rejected is
reported back with the reason, so a malformed rule fails loudly rather than
silently excluding nothing.

Pass instances (or equivalent plain dicts) to
:meth:`nucleus.NucleusClient.create_benchmark_evaluation_v2` via ``exclusion_rules``::

    client.create_benchmark_evaluation_v2(
        benchmark_id,
        model_run_id,
        exclusion_rules=[
            BoxAreaExclusionRule(scope="annotation", target="groundTruth", min=1024),
            LabelExclusionRule(scope="item", target="prediction", labels=["ignore"]),
            MetadataExclusionRule(key="is_dark", op="EQ", value=True),
        ],
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from .constants import (
    EXCLUSION_KEY_KEY,
    EXCLUSION_MAX_KEY,
    EXCLUSION_MIN_KEY,
    EXCLUSION_OP_KEY,
    EXCLUSION_TARGET_KEY,
    EXCLUSION_TYPE_BOX_AREA,
    EXCLUSION_TYPE_LABELS,
    EXCLUSION_TYPE_METADATA,
    EXCLUSION_VALUE_KEY,
    LABELS_KEY,
    SCOPE_KEY,
    TYPE_KEY,
)

# String literals are sent as values (not keys), so the server's request-body
# camelcaser preserves them verbatim — emit them exactly as the backend expects.
ExclusionScope = Literal["item", "annotation"]
ExclusionTarget = Literal["groundTruth", "prediction"]
MetadataOp = Literal["EQ", "IN", "GT", "LT"]


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
            TYPE_KEY: EXCLUSION_TYPE_METADATA,
            SCOPE_KEY: self.scope,
            EXCLUSION_KEY_KEY: self.key,
            EXCLUSION_OP_KEY: self.op,
            EXCLUSION_VALUE_KEY: self.value,
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
            TYPE_KEY: EXCLUSION_TYPE_LABELS,
            SCOPE_KEY: self.scope,
            EXCLUSION_TARGET_KEY: self.target,
            LABELS_KEY: list(self.labels),
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
            TYPE_KEY: EXCLUSION_TYPE_BOX_AREA,
            SCOPE_KEY: self.scope,
            EXCLUSION_TARGET_KEY: self.target,
        }
        if self.min is not None:
            out[EXCLUSION_MIN_KEY] = self.min
        if self.max is not None:
            out[EXCLUSION_MAX_KEY] = self.max
        return out


EvaluationV2ExclusionRule = Union[
    MetadataExclusionRule,
    LabelExclusionRule,
    BoxAreaExclusionRule,
]
