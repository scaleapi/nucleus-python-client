"""Benchmarks — frozen, cross-dataset ground-truth item sets for model evaluation.

A benchmark is a named collection of dataset items (with ground truth) that
model runs are evaluated against. Benchmark evaluations score every benchmark
item: items a model run has no predictions for count as false negatives, so
leaderboard scores stay comparable across runs with different coverage.

Create and manage benchmarks via :class:`~nucleus.NucleusClient`::

    benchmark = client.create_benchmark("city-streets-v1", slice_id="slc_...")
    evaluation = benchmark.create_evaluation_v2(
        model_run_id,
        rollup_groups=[RollupGroup("vehicle", ["car", "truck"])],
    )
    evaluation.wait_for_completion()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from nucleus.constants import (
    BENCHMARK_ID_KEY,
    CREATED_AT_KEY,
    CREATED_BY_USER_ID_KEY,
    DATASET_COUNT_KEY,
    DESCRIPTION_KEY,
    ITEM_COUNT_KEY,
    METADATA_KEY,
    NAME_KEY,
    SKIPPED_ITEMS_WITHOUT_GROUND_TRUTH_KEY,
)
from nucleus.data_transfer_object.evaluation_v2 import BenchmarkItemsPage
from nucleus.evaluation_v2 import (
    AllowedLabelMatch,
    EvaluationV2,
    RollupGroup,
)
from nucleus.evaluation_v2_exclusions import EvaluationV2ExclusionRule
from nucleus.evaluation_v2_preset import EvaluationV2Preset

if TYPE_CHECKING:
    from nucleus import NucleusClient


@dataclass
class Benchmark:
    """A benchmark: a frozen set of ground-truth items models are scored against."""

    id: str
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_by_user_id: Optional[str] = None
    created_at: Optional[str] = None
    item_count: Optional[int] = None
    dataset_count: Optional[int] = None
    skipped_items_without_ground_truth: Optional[int] = None
    _client: Optional["NucleusClient"] = field(repr=False, default=None)

    @classmethod
    def from_json(
        cls,
        payload: Dict[str, Any],
        client: Optional["NucleusClient"] = None,
    ) -> "Benchmark":
        return cls(
            id=str(payload[BENCHMARK_ID_KEY]),
            name=str(payload[NAME_KEY]),
            description=payload.get(DESCRIPTION_KEY),
            metadata=payload.get(METADATA_KEY),
            created_by_user_id=payload.get(CREATED_BY_USER_ID_KEY),
            created_at=payload.get(CREATED_AT_KEY),
            item_count=payload.get(ITEM_COUNT_KEY),
            dataset_count=payload.get(DATASET_COUNT_KEY),
            skipped_items_without_ground_truth=payload.get(
                SKIPPED_ITEMS_WITHOUT_GROUND_TRUTH_KEY
            ),
            _client=client,
        )

    def refresh(self) -> "Benchmark":
        """Reload this benchmark from Nucleus.

        Returns:
            self, with updated fields.
        """
        if self._client is None:
            raise RuntimeError(
                "Benchmark has no client; use NucleusClient.get_benchmark."
            )
        updated = self._client.get_benchmark(self.id)
        self.__dict__.update(updated.__dict__)
        return self

    def update(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Benchmark":
        """Update this benchmark's name, description, or metadata.

        Only the arguments you pass are changed. Benchmark membership is
        frozen at creation and cannot be updated.

        Returns:
            self, with updated fields.
        """
        if self._client is None:
            raise RuntimeError("Benchmark has no client.")
        updated = self._client.update_benchmark(
            self.id,
            name=name,
            description=description,
            metadata=metadata,
        )
        self.__dict__.update(updated.__dict__)
        return self

    def delete(self) -> None:
        """Delete this benchmark."""
        if self._client is None:
            raise RuntimeError("Benchmark has no client.")
        self._client.delete_benchmark(self.id)

    def items(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> BenchmarkItemsPage:
        """Return one page of this benchmark's member item ids.

        Parameters:
            limit: Optional page size.
            offset: Optional offset for pagination.

        Returns:
            :class:`~nucleus.data_transfer_object.evaluation_v2.BenchmarkItemsPage`:
            The page of dataset item ids and the total member count.
        """
        if self._client is None:
            raise RuntimeError("Benchmark has no client.")
        return self._client.list_benchmark_items(
            self.id, limit=limit, offset=offset
        )

    def create_evaluation_v2(
        self,
        model_run_id: str,
        *,
        name: Optional[str] = None,
        rollup_groups: Optional[List[RollupGroup]] = None,
        allowed_label_matches: Optional[List[AllowedLabelMatch]] = None,
        allowed_label_matches_id: Optional[str] = None,
        exclusion_rules: Optional[
            List[Union[EvaluationV2ExclusionRule, Dict[str, Any]]]
        ] = None,
        preset: Optional[EvaluationV2Preset] = None,
    ) -> EvaluationV2:
        """Evaluate a model run against this benchmark.

        See :meth:`NucleusClient.create_benchmark_evaluation_v2` for parameter
        details.

        Returns:
            :class:`~nucleus.evaluation_v2.EvaluationV2`: The created evaluation.
        """
        if self._client is None:
            raise RuntimeError("Benchmark has no client.")
        return self._client.create_benchmark_evaluation_v2(
            self.id,
            model_run_id,
            name=name,
            rollup_groups=rollup_groups,
            allowed_label_matches=allowed_label_matches,
            allowed_label_matches_id=allowed_label_matches_id,
            exclusion_rules=exclusion_rules,
            preset=preset,
        )
