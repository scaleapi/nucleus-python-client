import pytest

from nucleus import Dataset, DatasetItem, VideoScene
from nucleus.async_job import JobError
from nucleus.constants import (
    JOB_CREATION_TIME_KEY,
    JOB_ID_KEY,
    JOB_LAST_KNOWN_STATUS_KEY,
    JOB_TYPE_KEY,
)
from nucleus.deduplication import DeduplicationJob, DeduplicationResult
from nucleus.errors import NucleusAPIError

from .helpers import (
    DEDUP_DEFAULT_TEST_THRESHOLD,
    TEST_DATASET_ITEMS,
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    TEST_VIDEO_DATASET_NAME,
    TEST_VIDEO_SCENES,
    TEST_VIDEO_URL,
)


class FakeDeduplicationClient:
    def __init__(self):
        self.requests = []

    def make_request(self, payload, route, requests_command=None):
        self.requests.append(
            {
                "payload": payload,
                "route": route,
                "requests_command": requests_command,
            }
        )
        return {
            JOB_ID_KEY: "job_fake",
            JOB_LAST_KNOWN_STATUS_KEY: "Queued",
            JOB_TYPE_KEY: "deduplicate",
            JOB_CREATION_TIME_KEY: "2026-05-06T00:00:00Z",
        }


def _deduplication_result(job: DeduplicationJob) -> DeduplicationResult:
    assert isinstance(job, DeduplicationJob)
    result = job.result()
    assert isinstance(result, DeduplicationResult)
    return result


def test_deduplicate_empty_reference_ids_raises_error():
    fake_client = FakeDeduplicationClient()
    fake_dataset = Dataset("fake", fake_client)
    with pytest.raises(ValueError, match="reference_ids cannot be empty"):
        fake_dataset.deduplicate(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD, reference_ids=[]
        )
    assert fake_client.requests == []


def test_deduplicate_by_ids_empty_list_raises_error():
    fake_client = FakeDeduplicationClient()
    fake_dataset = Dataset("fake", fake_client)
    with pytest.raises(ValueError, match="dataset_item_ids must be non-empty"):
        fake_dataset.deduplicate_by_ids(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD, dataset_item_ids=[]
        )
    assert fake_client.requests == []


def test_deduplicate_defaults_to_async_route_and_returns_job():
    fake_client = FakeDeduplicationClient()
    fake_dataset = Dataset("fake", fake_client)

    job = fake_dataset.deduplicate(
        threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
        reference_ids=["ref_1", "ref_2"],
    )

    assert isinstance(job, DeduplicationJob)
    assert fake_client.requests == [
        {
            "payload": {
                "threshold": DEDUP_DEFAULT_TEST_THRESHOLD,
                "reference_ids": ["ref_1", "ref_2"],
            },
            "route": "dataset/fake/deduplicate_async",
            "requests_command": None,
        }
    ]


def test_deduplicate_by_ids_defaults_to_async_route_and_returns_job():
    fake_client = FakeDeduplicationClient()
    fake_dataset = Dataset("fake", fake_client)

    job = fake_dataset.deduplicate_by_ids(
        threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
        dataset_item_ids=["di_1", "di_2"],
    )

    assert isinstance(job, DeduplicationJob)
    assert fake_client.requests == [
        {
            "payload": {
                "dataset_item_ids": ["di_1", "di_2"],
                "threshold": DEDUP_DEFAULT_TEST_THRESHOLD,
            },
            "route": "dataset/fake/deduplicate_async",
            "requests_command": None,
        }
    ]


@pytest.fixture(scope="module")
def dataset_image_async(CLIENT):
    """Image dataset uploaded asynchronously."""
    ds = CLIENT.create_dataset(
        TEST_DATASET_NAME + " dedup async", is_scene=False
    )
    try:
        job = ds.append(TEST_DATASET_ITEMS, asynchronous=True)
        job.sleep_until_complete()
        yield ds
    finally:
        CLIENT.delete_dataset(ds.id)


@pytest.mark.integration
def test_deduplicate_image_async_entire_dataset(dataset_image_async):
    """Test deduplication on image dataset uploaded asynchronously."""
    result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )
    assert len(result.unique_reference_ids) > 0
    assert len(result.unique_item_ids) > 0
    assert result.stats.original_count == len(TEST_DATASET_ITEMS)


@pytest.mark.integration
def test_deduplicate_image_async_with_reference_ids(dataset_image_async):
    """Test deduplication with reference IDs on image dataset uploaded asynchronously."""
    reference_ids = [item.reference_id for item in TEST_DATASET_ITEMS[:2]]
    result = _deduplication_result(
        dataset_image_async.deduplicate(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
            reference_ids=reference_ids,
        )
    )
    assert result.stats.original_count == len(reference_ids)
    assert len(result.unique_reference_ids) <= len(reference_ids)
    assert len(result.unique_item_ids) <= len(reference_ids)


@pytest.mark.integration
def test_deduplicate_image_async_by_ids(dataset_image_async):
    """Test deduplicate_by_ids on image dataset uploaded asynchronously."""
    initial_result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )
    item_ids = initial_result.unique_item_ids
    assert len(item_ids) > 0

    result = _deduplication_result(
        dataset_image_async.deduplicate_by_ids(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
            dataset_item_ids=item_ids,
        )
    )
    assert result.stats.original_count == len(item_ids)
    assert result.unique_item_ids == initial_result.unique_item_ids


@pytest.fixture(scope="module")
def dataset_video_scene_async(CLIENT):
    """Video scene dataset (with frames) uploaded asynchronously."""
    ds = CLIENT.create_dataset(
        TEST_VIDEO_DATASET_NAME + " dedup async", is_scene=True
    )
    try:
        scene_1 = TEST_VIDEO_SCENES["scenes"][0]
        scenes = [VideoScene.from_json(scene_1)]
        job = ds.append(scenes, asynchronous=True)
        job.sleep_until_complete()
        yield ds
    finally:
        CLIENT.delete_dataset(ds.id)


def _get_scene_frame_ref_ids():
    """Extract frame reference IDs from TEST_VIDEO_SCENES scene_1."""
    return [
        frame["reference_id"]
        for frame in TEST_VIDEO_SCENES["scenes"][0]["frames"]
    ]


@pytest.mark.integration
def test_deduplicate_video_scene_async_entire_dataset(
    dataset_video_scene_async,
):
    """Test deduplication on video scene dataset uploaded asynchronously."""
    result = _deduplication_result(
        dataset_video_scene_async.deduplicate(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD
        )
    )
    assert len(result.unique_reference_ids) > 0
    assert len(result.unique_item_ids) > 0
    assert result.stats.original_count == len(_get_scene_frame_ref_ids())


@pytest.mark.integration
def test_deduplicate_video_scene_async_with_frame_reference_ids(
    dataset_video_scene_async,
):
    """Test deduplication with frame reference IDs on video scene dataset uploaded asynchronously."""
    frame_ref_ids = _get_scene_frame_ref_ids()
    result = _deduplication_result(
        dataset_video_scene_async.deduplicate(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
            reference_ids=frame_ref_ids,
        )
    )
    assert result.stats.original_count == len(frame_ref_ids)
    assert len(result.unique_reference_ids) <= len(frame_ref_ids)
    assert len(result.unique_item_ids) <= len(frame_ref_ids)


@pytest.mark.integration
def test_deduplicate_video_scene_async_by_ids(dataset_video_scene_async):
    """Test deduplicate_by_ids on video scene dataset uploaded asynchronously."""
    initial_result = _deduplication_result(
        dataset_video_scene_async.deduplicate(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD
        )
    )
    item_ids = initial_result.unique_item_ids
    assert len(item_ids) > 0

    result = _deduplication_result(
        dataset_video_scene_async.deduplicate_by_ids(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
            dataset_item_ids=item_ids,
        )
    )
    assert result.stats.original_count == len(item_ids)
    assert result.unique_item_ids == initial_result.unique_item_ids


@pytest.fixture(scope="module")
def dataset_video_url_async(CLIENT):
    """Video URL dataset uploaded asynchronously."""
    ds = CLIENT.create_dataset(
        TEST_VIDEO_DATASET_NAME + " video_url dedup async", is_scene=True
    )
    try:
        scene = VideoScene.from_json(
            {
                "reference_id": "video_url_scene_async",
                "video_url": TEST_VIDEO_URL,
                "metadata": {"test": "video_url_dedup_async"},
            }
        )
        job = ds.append([scene], asynchronous=True)
        job.sleep_until_complete()
        yield ds
    finally:
        CLIENT.delete_dataset(ds.id)


@pytest.mark.integration
def test_deduplicate_video_url_async_entire_dataset(dataset_video_url_async):
    """Test deduplication on video URL dataset uploaded asynchronously."""
    result = _deduplication_result(
        dataset_video_url_async.deduplicate(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD
        )
    )
    assert len(result.unique_reference_ids) > 0
    assert len(result.unique_item_ids) > 0
    assert result.stats.original_count > 0


@pytest.mark.integration
def test_deduplicate_video_url_async_by_ids(dataset_video_url_async):
    """Test deduplicate_by_ids on video URL dataset uploaded asynchronously."""
    initial_result = _deduplication_result(
        dataset_video_url_async.deduplicate(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD
        )
    )
    item_ids = initial_result.unique_item_ids
    assert len(item_ids) > 0

    result = _deduplication_result(
        dataset_video_url_async.deduplicate_by_ids(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
            dataset_item_ids=item_ids,
        )
    )
    assert result.stats.original_count == len(item_ids)
    assert result.unique_item_ids == initial_result.unique_item_ids


# Edge case tests


@pytest.mark.integration
def test_deduplicate_threshold_zero(dataset_image_async):
    """Verify threshold=0 (exact match only) succeeds and returns correct stats."""
    result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=0)
    )
    assert result.stats.threshold == 0


@pytest.mark.integration
def test_deduplicate_threshold_max(dataset_image_async):
    """Verify threshold=64 (maximum allowed) succeeds and returns correct stats."""
    result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=64)
    )
    assert result.stats.threshold == 64


@pytest.mark.integration
def test_deduplicate_threshold_negative(dataset_image_async):
    """Verify negative threshold raises an error (must be >= 0)."""
    with pytest.raises((NucleusAPIError, JobError)):
        _deduplication_result(dataset_image_async.deduplicate(threshold=-1))


@pytest.mark.integration
def test_deduplicate_threshold_too_high(dataset_image_async):
    """Verify threshold > 64 raises an error (must be <= 64)."""
    with pytest.raises((NucleusAPIError, JobError)):
        _deduplication_result(dataset_image_async.deduplicate(threshold=65))


@pytest.mark.integration
def test_deduplicate_threshold_non_integer(dataset_image_async):
    """Verify non-integer threshold raises an error."""
    with pytest.raises((NucleusAPIError, JobError)):
        _deduplication_result(dataset_image_async.deduplicate(threshold=10.5))


@pytest.mark.integration
def test_deduplicate_nonexistent_reference_id(dataset_image_async):
    """Verify nonexistent reference_id raises an error."""
    with pytest.raises((NucleusAPIError, JobError)):
        _deduplication_result(
            dataset_image_async.deduplicate(
                threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
                reference_ids=["nonexistent_ref_id"],
            )
        )


@pytest.mark.integration
def test_deduplicate_by_ids_nonexistent_id(dataset_image_async):
    """Verify nonexistent dataset_item_id raises an error."""
    with pytest.raises((NucleusAPIError, JobError)):
        _deduplication_result(
            dataset_image_async.deduplicate_by_ids(
                threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
                dataset_item_ids=["di_nonexistent"],
            )
        )


@pytest.mark.integration
def test_deduplicate_idempotency(dataset_image_async):
    """Verify repeated deduplication calls return consistent results."""
    result1 = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )
    result2 = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )

    assert result1.unique_item_ids == result2.unique_item_ids
    assert result1.unique_reference_ids == result2.unique_reference_ids
    assert result1.stats.original_count == result2.stats.original_count
    assert result1.stats.deduplicated_count == result2.stats.deduplicated_count


@pytest.mark.integration
def test_deduplicate_response_invariants(dataset_image_async):
    """Verify response maintains expected invariants between fields."""
    result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )

    assert len(result.unique_item_ids) == len(result.unique_reference_ids)
    assert result.stats.deduplicated_count == len(result.unique_item_ids)
    assert result.stats.deduplicated_count <= result.stats.original_count
    assert result.stats.threshold == DEDUP_DEFAULT_TEST_THRESHOLD


@pytest.mark.integration
def test_deduplicate_by_ids_threshold_negative(dataset_image_async):
    """Verify deduplicate_by_ids rejects negative threshold."""
    initial_result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )
    item_ids = initial_result.unique_item_ids

    with pytest.raises((NucleusAPIError, JobError)):
        _deduplication_result(
            dataset_image_async.deduplicate_by_ids(
                threshold=-1, dataset_item_ids=item_ids
            )
        )


@pytest.mark.integration
def test_deduplicate_by_ids_threshold_too_high(dataset_image_async):
    """Verify deduplicate_by_ids rejects threshold > 64."""
    initial_result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )
    item_ids = initial_result.unique_item_ids

    with pytest.raises((NucleusAPIError, JobError)):
        _deduplication_result(
            dataset_image_async.deduplicate_by_ids(
                threshold=65, dataset_item_ids=item_ids
            )
        )


@pytest.mark.integration
def test_deduplicate_single_item(dataset_image_async):
    """Verify single item deduplication returns that item as unique."""
    reference_ids = [TEST_DATASET_ITEMS[0].reference_id]
    result = _deduplication_result(
        dataset_image_async.deduplicate(
            threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
            reference_ids=reference_ids,
        )
    )

    assert result.stats.original_count == 1
    assert result.stats.deduplicated_count == 1
    assert len(result.unique_reference_ids) == 1


@pytest.fixture(scope="function")
def dataset_empty(CLIENT):
    """Empty dataset with no items."""
    ds = CLIENT.create_dataset(TEST_DATASET_NAME + " empty", is_scene=False)
    try:
        yield ds
    finally:
        CLIENT.delete_dataset(ds.id)


@pytest.mark.integration
def test_deduplicate_empty_dataset(dataset_empty):
    """Verify deduplication on empty dataset returns zero counts."""
    result = _deduplication_result(
        dataset_empty.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )

    assert result.stats.original_count == 0
    assert result.stats.deduplicated_count == 0
    assert len(result.unique_reference_ids) == 0
    assert len(result.unique_item_ids) == 0


@pytest.fixture(scope="function")
def dataset_with_duplicates(CLIENT):
    """Dataset with duplicate images (same image uploaded twice)."""
    ds = CLIENT.create_dataset(
        TEST_DATASET_NAME + " duplicates", is_scene=False
    )
    try:
        items = [
            DatasetItem(TEST_IMG_URLS[0], reference_id="img_original"),
            DatasetItem(TEST_IMG_URLS[0], reference_id="img_duplicate"),
            DatasetItem(TEST_IMG_URLS[1], reference_id="img_different"),
        ]
        ds.append(items)
        yield ds
    finally:
        CLIENT.delete_dataset(ds.id)


@pytest.mark.integration
def test_deduplicate_identifies_duplicates(dataset_with_duplicates):
    """Verify deduplication actually identifies duplicate images."""
    result = _deduplication_result(
        dataset_with_duplicates.deduplicate(threshold=0)
    )

    assert result.stats.original_count == 3
    # With threshold=0, the two identical images should be deduplicated to one
    assert result.stats.deduplicated_count == 2
    assert len(result.unique_reference_ids) == 2


@pytest.mark.integration
def test_deduplicate_distinct_images_all_unique(dataset_image_async):
    """Distinct images should all remain after deduplication."""
    result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=0)
    )

    # With threshold=0 (exact match only), all distinct images should be unique
    assert result.stats.deduplicated_count == result.stats.original_count


# ---------------------------------------------------------------------------
# Default async job tests
#
# The SDK kicks off a Temporal-backed dedup job on the server, returns a
# `DeduplicationJob`, and the caller polls/awaits via `job.result()`. The
# result payload should match the direct result contract for the same inputs.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_deduplicate_returns_job_and_result(dataset_image_async):
    """`deduplicate` returns a DeduplicationJob whose `.result()` blocks until
    the workflow completes and yields a DeduplicationResult."""
    first_result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )

    job = dataset_image_async.deduplicate(
        threshold=DEDUP_DEFAULT_TEST_THRESHOLD
    )
    assert isinstance(job, DeduplicationJob)
    assert (
        job.job_type
    )  # populated by AsyncJob.from_json from the server response

    result = job.result()
    assert isinstance(result, DeduplicationResult)
    assert result.stats.threshold == DEDUP_DEFAULT_TEST_THRESHOLD
    assert result.stats.original_count == first_result.stats.original_count
    assert (
        result.stats.deduplicated_count
        == first_result.stats.deduplicated_count
    )
    assert set(result.unique_reference_ids) == set(
        first_result.unique_reference_ids
    )


@pytest.mark.integration
def test_deduplicate_with_reference_ids_returns_job(dataset_image_async):
    """Default async mode respects an explicit `reference_ids` scope."""
    reference_ids = [item.reference_id for item in TEST_DATASET_ITEMS[:2]]

    job = dataset_image_async.deduplicate(
        threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
        reference_ids=reference_ids,
    )
    assert isinstance(job, DeduplicationJob)

    result = job.result()
    assert isinstance(result, DeduplicationResult)
    assert result.stats.original_count == len(reference_ids)
    assert len(result.unique_reference_ids) <= len(reference_ids)


@pytest.mark.integration
def test_deduplicate_by_ids_returns_job(dataset_image_async):
    """Default async `deduplicate_by_ids` returns a job and result."""
    initial_result = _deduplication_result(
        dataset_image_async.deduplicate(threshold=DEDUP_DEFAULT_TEST_THRESHOLD)
    )
    item_ids = initial_result.unique_item_ids
    assert len(item_ids) > 0

    job = dataset_image_async.deduplicate_by_ids(
        threshold=DEDUP_DEFAULT_TEST_THRESHOLD,
        dataset_item_ids=item_ids,
    )
    assert isinstance(job, DeduplicationJob)

    result = job.result()
    assert isinstance(result, DeduplicationResult)
    assert result.stats.original_count == len(item_ids)
    assert result.unique_item_ids == initial_result.unique_item_ids


@pytest.mark.integration
def test_deduplicate_identifies_duplicates_via_job(dataset_with_duplicates):
    """Sanity check: default async mode produces the expected dedup outcome."""
    job = dataset_with_duplicates.deduplicate(threshold=0)
    assert isinstance(job, DeduplicationJob)

    result = job.result()
    assert result.stats.original_count == 3
    assert result.stats.deduplicated_count == 2
    assert len(result.unique_reference_ids) == 2


@pytest.mark.integration
def test_deduplicate_split_wait_and_fetch(dataset_image_async):
    """`result(wait_for_completion=False)` is a usable mode when the caller
    has already blocked via `sleep_until_complete()`."""
    job = dataset_image_async.deduplicate(
        threshold=DEDUP_DEFAULT_TEST_THRESHOLD
    )
    assert isinstance(job, DeduplicationJob)

    job.sleep_until_complete(verbose_std_out=False)
    result = job.result(wait_for_completion=False)
    assert isinstance(result, DeduplicationResult)
    assert result.stats.threshold == DEDUP_DEFAULT_TEST_THRESHOLD
