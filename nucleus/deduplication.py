from dataclasses import dataclass
from typing import Any, Dict, List, cast

from nucleus.async_job import AsyncJob, JobError


@dataclass
class DeduplicationStats:
    """Summary statistics for a deduplication run.

    Attributes:
        threshold: The Hamming distance threshold the run was executed at.
            Lower values are stricter; ``0`` means exact matches only.
        original_count: How many items were considered before deduplication.
        deduplicated_count: How many unique items remained afterwards.
    """

    threshold: int
    original_count: int
    deduplicated_count: int


@dataclass
class DeduplicationResult:
    """Output of a deduplication run.

    Attributes:
        unique_item_ids: Nucleus-internal dataset item IDs (e.g.
            ``"di_abc123..."``) that survived deduplication. One entry per
            kept item.
        unique_reference_ids: The user-defined reference IDs you supplied at
            upload time, in the same order as ``unique_item_ids``.
        stats: Summary statistics for the run. See :class:`DeduplicationStats`.
    """

    unique_item_ids: List[str]
    unique_reference_ids: List[str]
    stats: DeduplicationStats


class DeduplicationJob(AsyncJob):
    """Handle to a long-running deduplication job.

    Returned from :meth:`Dataset.deduplicate` and
    :meth:`Dataset.deduplicate_by_ids`. Deduplication always runs in the
    background; collect the completed output with :meth:`result`.

    Inherits all the standard :class:`AsyncJob` controls
    (:meth:`status`, :meth:`errors`, :meth:`sleep_until_complete`).

    ::

        import nucleus

        client = nucleus.NucleusClient(YOUR_API_KEY)
        dataset = client.get_dataset("ds_xxx")

        job = dataset.deduplicate(threshold=10)
        result = job.result()              # blocks until done
        print(result.stats.deduplicated_count)
        print(result.unique_reference_ids)

        # You can also deduplicate a known set of internal dataset item IDs.
        job = dataset.deduplicate_by_ids(
            threshold=10,
            dataset_item_ids=["di_xxx", "di_yyy"],
        )
        result = job.result()

        # Or split the wait and fetch yourself.
        job.sleep_until_complete()
        result = job.result(wait_for_completion=False)
    """

    def result(
        self, wait_for_completion: bool = True
    ) -> "DeduplicationResult":
        """Return the deduplication result, optionally waiting for the job.

        Parameters:
            wait_for_completion: When ``True`` (default), block until the job
                reaches a terminal state. When ``False``, the caller is
                expected to have already waited (e.g. via
                :meth:`sleep_until_complete`).

        Returns:
            A :class:`DeduplicationResult` containing the kept item IDs,
            reference IDs, and run statistics.

        Raises:
            JobError: If the job did not finish successfully (e.g. it was
                cancelled or hit a server error).
        """
        if wait_for_completion:
            self.sleep_until_complete(verbose_std_out=False)

        status = self.status()
        if status["status"] != "Completed":
            raise JobError(status, self)

        # AsyncJob.status() is typed as Dict[str, str] in the base class, but
        # the `message` slot is a JSON dict in practice. Cast locally so
        # static checkers don't flag the dict accesses below.
        msg = cast(Dict[str, Any], status["message"] or {})
        stats = cast(Dict[str, Any], msg.get("stats") or {})
        return DeduplicationResult(
            unique_item_ids=msg["unique_item_ids"],
            unique_reference_ids=msg["unique_reference_ids"],
            stats=DeduplicationStats(
                threshold=stats.get("threshold", 0),
                original_count=stats["original_count"],
                deduplicated_count=stats["deduplicated_count"],
            ),
        )
