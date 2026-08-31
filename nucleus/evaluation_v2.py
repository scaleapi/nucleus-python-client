"""Evaluation V2 — metrics and examples for a model run."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import requests

from nucleus.constants import (
    ALLOWED_LABEL_MATCHES_ID_KEY,
    ALLOWED_LABEL_MATCHES_KEY,
    ALLOWED_LABEL_MATCHES_NAME_KEY,
    BENCHMARK_ID_KEY,
    CLASS_NAME_CAMEL_KEY,
    CLASS_NAME_KEY,
    CREATED_AT_KEY,
    ERROR_MESSAGE_KEY,
    EVALUATION_ID_KEY,
    EXCLUSION_RULES_KEY,
    EXCLUSION_STATS_KEY,
    FILTERS_KEY,
    GROUND_TRUTH_LABEL_CAMEL_KEY,
    GROUND_TRUTH_LABEL_KEY,
    ID_KEY,
    IOU_THRESHOLD_KEY,
    LABELS_KEY,
    LIMIT_KEY,
    MATCH_TYPE_KEY,
    MODEL_ID_KEY,
    MODEL_PREDICTION_LABEL_CAMEL_KEY,
    MODEL_PREDICTION_LABEL_KEY,
    MODEL_RUN_ID_KEY,
    NAME_KEY,
    OFFSET_KEY,
    QUERY_KEY,
    ROLLUP_GROUPS_KEY,
    SLICE_ID_KEY,
    SORT_BY_KEY,
    SORT_ORDER_KEY,
    STATUS_KEY,
    TEMPORAL_WORKFLOW_ID_KEY,
)
from nucleus.data_transfer_object.evaluation_v2 import (
    EvaluationV2Charts,
    EvaluationV2ExamplesPage,
    EvaluationV2FilterArgs,
    EvaluationV2FilterSchema,
)

if TYPE_CHECKING:
    from nucleus import NucleusClient


class EvaluationV2Status(str, Enum):
    """Status of an Evaluation V2 run."""

    PENDING = "pending"
    COMPUTING = "computing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_OK: Set[EvaluationV2Status] = {
    EvaluationV2Status.SUCCEEDED,
    EvaluationV2Status.CANCELLED,
}

_ALLOWED_LABEL_MATCHES_DEPRECATION = (
    "allowed_label_matches is deprecated and will be removed in a future "
    "release. Use rollup_groups instead."
)


def _warn_allowed_label_matches_deprecated() -> None:
    """Emit the Evaluation V2 ``allowed_label_matches`` deprecation warning.

    ``stacklevel=3`` points at the public method's caller (this helper →
    the client/wrapper method → user code).
    """
    warnings.warn(
        _ALLOWED_LABEL_MATCHES_DEPRECATION,
        DeprecationWarning,
        stacklevel=3,
    )


def _parse_json_field(value: Any) -> Optional[Any]:
    """Normalize a field that may arrive already decoded or as a JSON string."""
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value


@dataclass
class AllowedLabelMatch:
    """Deprecated. Use :class:`RollupGroup` instead.

    Ground-truth and prediction label pair that counts as a match.
    """

    ground_truth_label: str
    model_prediction_label: str

    def to_api_dict(self) -> Dict[str, str]:
        return {
            GROUND_TRUTH_LABEL_KEY: self.ground_truth_label,
            MODEL_PREDICTION_LABEL_KEY: self.model_prediction_label,
        }


def _parse_allowed_label_matches(
    raw_matches: Any,
) -> Optional[List[AllowedLabelMatch]]:
    """Parse an ``allowed_label_matches`` array from an API payload.

    Tolerates either key casing and drops malformed entries.
    """
    if not isinstance(raw_matches, list):
        return None
    matches: List[AllowedLabelMatch] = []
    for m in raw_matches:
        if not isinstance(m, dict):
            continue
        gt = m.get(GROUND_TRUTH_LABEL_CAMEL_KEY)
        if gt is None:
            gt = m.get(GROUND_TRUTH_LABEL_KEY)
        mp = m.get(MODEL_PREDICTION_LABEL_CAMEL_KEY)
        if mp is None:
            mp = m.get(MODEL_PREDICTION_LABEL_KEY)
        if gt is not None and mp is not None:
            matches.append(
                AllowedLabelMatch(
                    ground_truth_label=str(gt),
                    model_prediction_label=str(mp),
                )
            )
    return matches


@dataclass
class RollupGroup:
    """A rollup class: raw labels evaluated together under one class name.

    Rollup groups are the primary label configuration for benchmark
    evaluations — each group maps a set of raw ground-truth/prediction
    labels onto a single canonical ``class_name``. A label may appear in
    at most one group across the configuration.
    """

    class_name: str
    labels: List[str]

    def to_api_dict(self) -> Dict[str, Any]:
        return {CLASS_NAME_KEY: self.class_name, LABELS_KEY: list(self.labels)}


def _parse_rollup_groups(raw_groups: Any) -> Optional[List[RollupGroup]]:
    """Parse a ``rollup_groups`` array from an API payload.

    Tolerates either key casing and drops malformed entries.
    """
    if not isinstance(raw_groups, list):
        return None
    groups: List[RollupGroup] = []
    for g in raw_groups:
        if not isinstance(g, dict):
            continue
        class_name = g.get(CLASS_NAME_CAMEL_KEY)
        if class_name is None:
            class_name = g.get(CLASS_NAME_KEY)
        labels = g.get(LABELS_KEY)
        if class_name is not None and isinstance(labels, list):
            groups.append(
                RollupGroup(
                    class_name=str(class_name),
                    labels=[str(label) for label in labels],
                )
            )
    return groups


@dataclass
class EvaluationV2:
    """An Evaluation V2 run for a model run or a run-free model."""

    id: str
    model_run_id: Optional[str]
    status: str
    model_id: Optional[str] = None
    name: Optional[str] = None
    temporal_workflow_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    #: Deprecated. Prefer :attr:`rollup_groups`.
    allowed_label_matches_id: Optional[str] = None
    #: Deprecated. Prefer :attr:`rollup_groups`.
    allowed_label_matches: Optional[List[AllowedLabelMatch]] = None
    #: Deprecated. Prefer :attr:`rollup_groups`.
    allowed_label_matches_name: Optional[str] = None
    rollup_groups: Optional[List[RollupGroup]] = None
    benchmark_id: Optional[str] = None
    slice_id: Optional[str] = None
    exclusion_rules: Optional[List[Dict[str, Any]]] = None
    exclusion_stats: Optional[Dict[str, Any]] = None
    _client: Optional["NucleusClient"] = field(repr=False, default=None)

    @classmethod
    def from_json(
        cls,
        payload: Dict[str, Any],
        client: Optional["NucleusClient"] = None,
    ) -> "EvaluationV2":
        matches = _parse_allowed_label_matches(
            payload.get(ALLOWED_LABEL_MATCHES_KEY)
        )

        return cls(
            id=str(payload[ID_KEY]),
            model_run_id=(
                str(payload[MODEL_RUN_ID_KEY])
                if payload.get(MODEL_RUN_ID_KEY) is not None
                else None
            ),
            status=str(payload[STATUS_KEY]),
            model_id=(
                str(payload[MODEL_ID_KEY])
                if payload.get(MODEL_ID_KEY) is not None
                else None
            ),
            name=payload.get(NAME_KEY),
            temporal_workflow_id=payload.get(TEMPORAL_WORKFLOW_ID_KEY),
            error_message=payload.get(ERROR_MESSAGE_KEY),
            created_at=payload.get(CREATED_AT_KEY),
            allowed_label_matches_id=payload.get(ALLOWED_LABEL_MATCHES_ID_KEY),
            allowed_label_matches=matches,
            allowed_label_matches_name=payload.get(
                ALLOWED_LABEL_MATCHES_NAME_KEY
            ),
            rollup_groups=_parse_rollup_groups(
                _parse_json_field(payload.get(ROLLUP_GROUPS_KEY))
            ),
            benchmark_id=payload.get(BENCHMARK_ID_KEY),
            slice_id=payload.get(SLICE_ID_KEY),
            exclusion_rules=_parse_json_field(
                payload.get(EXCLUSION_RULES_KEY)
            ),
            exclusion_stats=_parse_json_field(
                payload.get(EXCLUSION_STATS_KEY)
            ),
            _client=client,
        )

    def refresh(self) -> "EvaluationV2":
        """Reload this evaluation from Nucleus.

        Returns:
            self, with updated fields.
        """
        if self._client is None:
            raise RuntimeError(
                "EvaluationV2 has no client; use NucleusClient.get_evaluation_v2."
            )
        data = self._client.get(f"evaluationsV2/{self.id}")
        updated = EvaluationV2.from_json(data, self._client)
        self.__dict__.update(updated.__dict__)
        return self

    def wait_for_completion(
        self,
        timeout_sec: float = 600,
        poll_interval: float = 5,
    ) -> "EvaluationV2":
        """Wait until the evaluation finishes or is cancelled.

        Parameters:
            timeout_sec: Maximum seconds to wait.
            poll_interval: Seconds between status checks.

        Returns:
            self, after a terminal status is reached.

        Raises:
            RuntimeError: If the evaluation fails or times out.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.refresh()
            if self.status == EvaluationV2Status.FAILED:
                raise RuntimeError(
                    f"Evaluation {self.id} failed: {self.error_message or 'unknown'}"
                )
            if self.status in _TERMINAL_OK:
                return self
            time.sleep(poll_interval)
        raise RuntimeError(
            f"Timed out after {timeout_sec}s waiting for evaluation {self.id} "
            f"(last status: {self.status})"
        )

    def delete(self) -> None:
        """Delete this evaluation."""
        if self._client is None:
            raise RuntimeError("EvaluationV2 has no client.")
        self._client.make_request(
            {},
            f"evaluationsV2/{self.id}",
            requests_command=requests.delete,
            return_raw_response=True,
        )

    def cancel(self) -> "EvaluationV2":
        """Cancel this evaluation if it is still running.

        Stops the evaluation and sets its status to ``cancelled``. Finished
        evaluations cannot be cancelled (use :meth:`delete` to archive them).

        Returns:
            self, refreshed with the post-cancel status.
        """
        if self._client is None:
            raise RuntimeError("EvaluationV2 has no client.")
        self._client.make_request(
            {},
            f"evaluationsV2/{self.id}/cancel",
            requests_command=requests.post,
            return_raw_response=True,
        )
        return self.refresh()

    def retry(self) -> "EvaluationV2":
        """Retry this evaluation if it failed.

        Creates a new evaluation for the same model run, reusing this
        evaluation's slice, rollup groups, and exclusion rules. Only
        ``failed`` evaluations can be retried.

        Returns:
            :class:`EvaluationV2`: The newly created (retry) evaluation.
        """
        if self._client is None:
            raise RuntimeError("EvaluationV2 has no client.")
        result = self._client.post({}, f"evaluationsV2/{self.id}/retry")
        eval_id = result.get(EVALUATION_ID_KEY)
        if not eval_id:
            raise RuntimeError(
                f"Unexpected retry evaluation V2 response: {result}"
            )
        return self._client.get_evaluation_v2(str(eval_id))

    def charts(
        self,
        iou_threshold: float = 0.5,
        filters: Optional[
            Union[EvaluationV2FilterArgs, Dict[str, Any]]
        ] = None,
        query: Optional[str] = None,
    ) -> EvaluationV2Charts:
        """Return aggregate metrics for this evaluation.

        Parameters:
            iou_threshold: IoU threshold for matching (default 0.5).
            filters: Optional filters (:class:`EvaluationV2FilterArgs` or dict).
            query: Optional query string to narrow results.

        Returns:
            :class:`EvaluationV2Charts`: Summary metrics (mAP, confusion matrix, PR curve, etc.).
        """
        if self._client is None:
            raise RuntimeError("EvaluationV2 has no client.")
        payload: Dict[str, Any] = {IOU_THRESHOLD_KEY: iou_threshold}
        if filters is not None:
            if isinstance(filters, EvaluationV2FilterArgs):
                payload[FILTERS_KEY] = filters.to_api_filters()
            else:
                payload[FILTERS_KEY] = filters
        if query:
            payload[QUERY_KEY] = query
        data = self._client.post(payload, f"evaluationsV2/{self.id}/charts")
        return EvaluationV2Charts.parse_obj(data)

    def filter_schema(self) -> EvaluationV2FilterSchema:
        """Return the filter vocabulary for this evaluation.

        Lists the ground-truth labels, prediction labels, and item-metadata
        fields (with inferred value types and distinct values) present in this
        evaluation's results — the valid inputs for
        :class:`~nucleus.data_transfer_object.evaluation_v2.EvaluationV2FilterArgs`
        when calling :meth:`charts` or :meth:`examples`.

        Returns:
            :class:`~nucleus.data_transfer_object.evaluation_v2.EvaluationV2FilterSchema`.
        """
        if self._client is None:
            raise RuntimeError("EvaluationV2 has no client.")
        data = self._client.get(f"evaluationsV2/{self.id}/filterSchema")
        return EvaluationV2FilterSchema.parse_obj(data)

    def examples(
        self,
        match_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        filters: Optional[
            Union[EvaluationV2FilterArgs, Dict[str, Any]]
        ] = None,
        query: Optional[str] = None,
    ) -> EvaluationV2ExamplesPage:
        """Return paginated match examples, optionally filtered by match type.

        Parameters:
            match_type: ``"TP"``, ``"FP"``, or ``"FN"``. Omit (or ``None``) to
                return examples of all match types.
            limit: Page size (default 50, max 100).
            offset: Offset for pagination.
            sort_by: Optional field to sort by — one of ``"confidence"``,
                ``"iou"``, ``"dataset_item_id"``, ``"gt_area"``.
            sort_order: Optional sort direction (``"ASC"`` or ``"DESC"``).
            filters: Optional filters (:class:`EvaluationV2FilterArgs` or dict).
            query: Optional query string to narrow results.

        Returns:
            :class:`EvaluationV2ExamplesPage`: Matching rows and total count.
        """
        if self._client is None:
            raise RuntimeError("EvaluationV2 has no client.")
        payload: Dict[str, Any] = {
            LIMIT_KEY: limit,
            OFFSET_KEY: offset,
        }
        if match_type is not None:
            payload[MATCH_TYPE_KEY] = match_type
        if sort_by is not None:
            payload[SORT_BY_KEY] = sort_by
        if sort_order is not None:
            payload[SORT_ORDER_KEY] = sort_order
        if filters is not None:
            if isinstance(filters, EvaluationV2FilterArgs):
                payload[FILTERS_KEY] = filters.to_api_filters()
            else:
                payload[FILTERS_KEY] = filters
        if query:
            payload[QUERY_KEY] = query
        data = self._client.post(payload, f"evaluationsV2/{self.id}/examples")
        return EvaluationV2ExamplesPage.parse_obj(data)
