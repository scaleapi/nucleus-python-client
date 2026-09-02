"""Benchmarks — cross-dataset ground-truth item sets for model evaluation.

A benchmark is a named collection of dataset items (with ground truth) that
model runs are evaluated against. Benchmark evaluations score every benchmark
item: items a model run has no predictions for count as false negatives, so
leaderboard scores stay comparable across runs with different coverage.

Create and manage benchmarks via :class:`~nucleus.NucleusClient`::

    benchmark = client.create_benchmark("city-streets-v1", slice_id="slc_...")
    evaluation = benchmark.create_evaluation_v2(
        model_id=model.id,
        rollup_groups=[RollupGroup("vehicle", ["car", "truck"])],
    )
    evaluation.wait_for_completion()

A finalized benchmark is immutable. To evolve one, create a **new version**
downstream of it with ``parent_benchmark_id`` (see
:meth:`NucleusClient.create_benchmark`) — the child inherits the parent's items,
adds/removes on top, and takes a minor (default) or major version bump.

To assemble a benchmark incrementally, create a **draft** (``draft=True``): add
items across many calls with :meth:`Benchmark.add_items` / remove with
:meth:`Benchmark.remove_items`, then :meth:`Benchmark.finalize` to freeze it into
a ``"ready"`` benchmark. A draft cannot be evaluated until finalized.
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
    PARENT_BENCHMARK_ID_KEY,
    SKIPPED_ITEMS_WITHOUT_GROUND_TRUTH_KEY,
    STATUS_KEY,
    VERSION_LABEL_KEY,
    VERSION_MAJOR_KEY,
    VERSION_MINOR_KEY,
)
from nucleus.data_transfer_object.evaluation_v2 import BenchmarkItemsPage
from nucleus.evaluation_v2 import (
    EvaluationV2,
    RollupGroup,
)
from nucleus.evaluation_v2_exclusions import EvaluationV2ExclusionRule
from nucleus.evaluation_v2_preset import EvaluationV2Preset

if TYPE_CHECKING:
    from nucleus import NucleusClient
    from nucleus.model import Model


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
    #: Lifecycle status: ``"draft"`` (mutable, not evaluable), ``"building"``
    #: (one-shot create still streaming members in), ``"ready"`` (immutable), or
    #: ``"failed"``.
    status: Optional[str] = None
    #: Lineage: this benchmark's parent version (``None`` for a root benchmark).
    parent_benchmark_id: Optional[str] = None
    #: Version of this benchmark relative to its lineage root (root defaults to 1.0).
    version_major: Optional[int] = None
    version_minor: Optional[int] = None
    #: Optional human-readable version label (e.g. ``"rc1"``, ``"holdout-v2"``).
    version_label: Optional[str] = None
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
            status=payload.get(STATUS_KEY),
            parent_benchmark_id=payload.get(PARENT_BENCHMARK_ID_KEY),
            version_major=payload.get(VERSION_MAJOR_KEY),
            version_minor=payload.get(VERSION_MINOR_KEY),
            version_label=payload.get(VERSION_LABEL_KEY),
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
        model_run_id: Optional[str] = None,
        *,
        model_id: Optional[Union[str, "Model"]] = None,
        name: Optional[str] = None,
        rollup_groups: Optional[List[RollupGroup]] = None,
        exclusion_rules: Optional[
            List[Union[EvaluationV2ExclusionRule, Dict[str, Any]]]
        ] = None,
        preset: Optional[EvaluationV2Preset] = None,
    ) -> EvaluationV2:
        """Evaluate a model against this benchmark.

        Anchor the evaluation on a model (``model_id``) for the run-free
        "model v2" flow. The legacy ``model_run_id`` anchor is deprecated.
        Provide exactly one. Uncovered benchmark members are scored as false
        negatives, so partial coverage still ranks comparably.

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
            model_id=model_id,
            name=name,
            rollup_groups=rollup_groups,
            exclusion_rules=exclusion_rules,
            preset=preset,
        )

    def add_items(
        self,
        *,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        scene_ids: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> "Benchmark":
        """Add items to this **draft** benchmark.

        Only valid while this benchmark is a draft (``status == "draft"``); a
        finalized benchmark is immutable. See
        :meth:`NucleusClient.add_benchmark_items` for parameter details.

        Returns:
            self, refreshed.
        """
        if self._client is None:
            raise RuntimeError("Benchmark has no client.")
        self._client.add_benchmark_items(
            self.id,
            item_ids=item_ids,
            items=items,
            slice_id=slice_id,
            dataset_id=dataset_id,
            slice_ids=slice_ids,
            dataset_ids=dataset_ids,
            scene_ids=scene_ids,
            wait_for_completion=wait_for_completion,
            verbose=verbose,
        )
        return self.refresh()

    def remove_items(self, item_ids: List[str]) -> "Benchmark":
        """Remove items from this **draft** benchmark.

        Only valid while this benchmark is a draft. Unknown ids are ignored.

        Parameters:
            item_ids: Dataset item ids (``di_*``) to remove.

        Returns:
            self, refreshed.
        """
        if self._client is None:
            raise RuntimeError("Benchmark has no client.")
        self._client.remove_benchmark_items(self.id, item_ids)
        return self.refresh()

    def finalize(self) -> "Benchmark":
        """Finalize this **draft** benchmark, freezing it into a ``"ready"`` one.

        After finalizing, the benchmark is immutable and can be evaluated. Fails
        if it is not a draft, is empty, or still has an add-items job in flight.

        Returns:
            self, with updated fields (``status`` now ``"ready"``).
        """
        if self._client is None:
            raise RuntimeError("Benchmark has no client.")
        updated = self._client.finalize_benchmark(self.id)
        self.__dict__.update(updated.__dict__)
        return self
