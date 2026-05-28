"""Evaluation V2 — metrics and examples for a model run."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union
from urllib.parse import urlencode

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
    _client: Optional["NucleusClient"] = field(repr=False, default=None)

    @classmethod
    def from_json(
        cls,
        payload: Dict[str, Any],
        client: Optional["NucleusClient"] = None,
    ) -> "EvaluationV2":
        raw_matches = payload.get("allowed_label_matches")
        matches: Optional[List[AllowedLabelMatch]] = None
        if isinstance(raw_matches, list):
            matches = []
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
        params: Dict[str, str] = {}
        params["iouThreshold"] = str(iou_threshold)
        if filters is not None:
            if isinstance(filters, EvaluationV2FilterArgs):
                filt_dict = filters.to_api_filters()
            else:
                filt_dict = filters
            params["filters"] = json.dumps(filt_dict)
        if query:
            params["query"] = query
        qs = urlencode(params)
        route = f"evaluationsV2/{self.id}/charts?{qs}"
        data = self._client.get(route)
        return EvaluationV2Charts.parse_obj(data)

    def examples(
        self,
        match_type: str,
        limit: int = 50,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        filters: Optional[
            Union[EvaluationV2FilterArgs, Dict[str, Any]]
        ] = None,
        query: Optional[str] = None,
    ) -> EvaluationV2ExamplesPage:
        """Return paginated true-positive, false-positive, or false-negative examples.

        Parameters:
            match_type: ``"TP"``, ``"FP"``, or ``"FN"``.
            limit: Page size (default 50).
            offset: Row offset for pagination.
            sort_by: Optional field to sort by.
            sort_order: Optional sort direction (e.g. ``"asc"`` or ``"desc"``).
            filters: Optional filters (:class:`EvaluationV2FilterArgs` or dict).
            query: Optional query string to narrow results.

        Returns:
            :class:`EvaluationV2ExamplesPage`: Matching rows and total count.
        """
        if self._client is None:
            raise RuntimeError("EvaluationV2 has no client.")
        payload: Dict[str, Any] = {
            "match_type": match_type,
            "limit": limit,
            "offset": offset,
        }
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
