"""Evaluation V2 — metrics and examples for a model run."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import requests

from nucleus.data_transfer_object.evaluation_v2 import (
    EvaluationV2Charts,
    EvaluationV2ExamplesPage,
    EvaluationV2FilterArgs,
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


def _parse_json_field(value: Any) -> Optional[Any]:
    """Normalize a JSONB column that may arrive as a string or already parsed.

    The REST ``GET``/``LIST`` evaluation endpoints return raw DB rows, so the
    ``exclusion_rules`` / ``exclusion_stats`` jsonb columns can come back either
    already decoded (dict/list) or as a JSON string depending on the driver.
    """
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
    """Ground-truth and prediction label pair that counts as a match."""

    ground_truth_label: str
    model_prediction_label: str

    def to_api_dict(self) -> Dict[str, str]:
        return {
            "ground_truth_label": self.ground_truth_label,
            "model_prediction_label": self.model_prediction_label,
        }


def parse_allowed_label_matches(
    raw_matches: Any,
) -> Optional[List[AllowedLabelMatch]]:
    """Parse an ``allowed_label_matches`` array from an API payload.

    Accepts both the camelCase (``groundTruthLabel`` / ``modelPredictionLabel``)
    and snake_case shapes the backend may return, and drops malformed entries.
    """
    if not isinstance(raw_matches, list):
        return None
    matches: List[AllowedLabelMatch] = []
    for m in raw_matches:
        if not isinstance(m, dict):
            continue
        gt = m.get("groundTruthLabel")
        if gt is None:
            gt = m.get("ground_truth_label")
        mp = m.get("modelPredictionLabel")
        if mp is None:
            mp = m.get("model_prediction_label")
        if gt is not None and mp is not None:
            matches.append(
                AllowedLabelMatch(
                    ground_truth_label=str(gt),
                    model_prediction_label=str(mp),
                )
            )
    return matches


@dataclass
class BatchEvaluationResult:
    """Outcome of one job in a batch create call.

    ``evaluation`` is set on success; ``error`` holds the error message on
    failure. Use :attr:`succeeded` to filter, and re-run the failed jobs by
    feeding their ``model_run_id`` / ``slice_id`` back into a new batch call.
    """

    model_run_id: str
    slice_id: Optional[str] = None
    name: Optional[str] = None
    evaluation: Optional["EvaluationV2"] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.evaluation is not None


@dataclass
class EvaluationV2:
    """An Evaluation V2 run for a model run."""

    id: str
    model_run_id: str
    dataset_id: str
    status: str
    name: Optional[str] = None
    temporal_workflow_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    allowed_label_matches_id: Optional[str] = None
    allowed_label_matches: Optional[List[AllowedLabelMatch]] = None
    allowed_label_matches_name: Optional[str] = None
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
        matches = parse_allowed_label_matches(
            payload.get("allowed_label_matches")
        )

        return cls(
            id=str(payload["id"]),
            model_run_id=str(payload["model_run_id"]),
            dataset_id=str(payload["dataset_id"]),
            status=str(payload["status"]),
            name=payload.get("name"),
            temporal_workflow_id=payload.get("temporal_workflow_id"),
            error_message=payload.get("error_message"),
            created_at=payload.get("created_at"),
            allowed_label_matches_id=payload.get("allowed_label_matches_id"),
            allowed_label_matches=matches,
            allowed_label_matches_name=payload.get(
                "allowed_label_matches_name"
            ),
            slice_id=payload.get("slice_id"),
            exclusion_rules=_parse_json_field(payload.get("exclusion_rules")),
            exclusion_stats=_parse_json_field(payload.get("exclusion_stats")),
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
        evaluation's slice, allowed-label-matches, and exclusion rules. Only
        ``failed`` evaluations can be retried.

        Returns:
            :class:`EvaluationV2`: The newly created (retry) evaluation.
        """
        if self._client is None:
            raise RuntimeError("EvaluationV2 has no client.")
        result = self._client.post({}, f"evaluationsV2/{self.id}/retry")
        eval_id = result.get("evaluation_id")
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
        payload: Dict[str, Any] = {"iouThreshold": iou_threshold}
        if filters is not None:
            if isinstance(filters, EvaluationV2FilterArgs):
                payload["filters"] = filters.to_api_filters()
            else:
                payload["filters"] = filters
        if query:
            payload["query"] = query
        data = self._client.post(payload, f"evaluationsV2/{self.id}/charts")
        return EvaluationV2Charts.parse_obj(data)

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
            offset: Row offset for pagination.
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
            "limit": limit,
            "offset": offset,
        }
        if match_type is not None:
            payload["match_type"] = match_type
        if sort_by is not None:
            payload["sort_by"] = sort_by
        if sort_order is not None:
            payload["sort_order"] = sort_order
        if filters is not None:
            if isinstance(filters, EvaluationV2FilterArgs):
                payload["filters"] = filters.to_api_filters()
            else:
                payload["filters"] = filters
        if query:
            payload["query"] = query
        data = self._client.post(payload, f"evaluationsV2/{self.id}/examples")
        return EvaluationV2ExamplesPage.parse_obj(data)
