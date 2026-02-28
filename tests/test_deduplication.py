import pytest

from nucleus import Dataset, DatasetItem, NucleusClient, VideoScene
from nucleus.deduplication import DeduplicationResult
from nucleus.errors import NucleusAPIError

from .helpers import (
    TEST_DATASET_ITEMS,
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    TEST_VIDEO_DATASET_NAME,
    TEST_VIDEO_SCENES,
)


def test_deduplicate_empty_reference_ids_raises_error():
    fake_dataset = Dataset("fake", NucleusClient("fake"))
    with pytest.raises(ValueError, match="reference_ids cannot be empty"):
        fake_dataset.deduplicate(threshold=10, reference_ids=[])


def test_deduplicate_by_ids_empty_list_raises_error():
    fake_dataset = Dataset("fake", NucleusClient("fake"))
    with pytest.raises(ValueError, match="dataset_item_ids must be non-empty"):
        fake_dataset.deduplicate_by_ids(threshold=10, dataset_item_ids=[])


@pytest.fixture(scope="module")
def dataset_image(CLIENT):
    """Image dataset with TEST_DATASET_ITEMS (waits for phash calculation)."""
    ds = CLIENT.create_dataset(TEST_DATASET_NAME + " dedup", is_scene=False)
    job = ds.append(TEST_DATASET_ITEMS, asynchronous=True)
    job.sleep_until_complete()
    yield ds
    CLIENT.delete_dataset(ds.id)


@pytest.mark.integration
def test_deduplicate_entire_dataset(dataset_image):
    result = dataset_image.deduplicate(threshold=10)
    assert isinstance(result, DeduplicationResult)
    assert len(result.unique_reference_ids) > 0
    assert len(result.unique_item_ids) > 0
    assert result.stats.original_count == len(TEST_DATASET_ITEMS)


@pytest.mark.integration
def test_deduplicate_with_reference_ids(dataset_image):
    reference_ids = [item.reference_id for item in TEST_DATASET_ITEMS[:2]]
    result = dataset_image.deduplicate(threshold=10, reference_ids=reference_ids)
    assert isinstance(result, DeduplicationResult)
    assert result.stats.original_count == len(reference_ids)
    assert len(result.unique_reference_ids) <= len(reference_ids)
    assert len(result.unique_item_ids) <= len(reference_ids)


@pytest.mark.integration
def test_deduplicate_by_ids(dataset_image):
    initial_result = dataset_image.deduplicate(threshold=10)
    item_ids = initial_result.unique_item_ids
    assert len(item_ids) > 0

    result = dataset_image.deduplicate_by_ids(threshold=10, dataset_item_ids=item_ids)
    assert isinstance(result, DeduplicationResult)
    assert result.stats.original_count == len(item_ids)
    assert result.unique_item_ids == initial_result.unique_item_ids


@pytest.fixture(scope="module")
def dataset_video_scene(CLIENT):
    """Scene dataset with scene_1 (frame IDs: video_frame_0, video_frame_1)."""
    ds = CLIENT.create_dataset(TEST_VIDEO_DATASET_NAME + " dedup", is_scene=True)
    scene_1 = TEST_VIDEO_SCENES["scenes"][0]
    scenes = [VideoScene.from_json(scene_1)]
    job = ds.append(scenes, asynchronous=True)
    job.sleep_until_complete()
    yield ds
    CLIENT.delete_dataset(ds.id)


def _get_scene_frame_ref_ids():
    """Extract frame reference IDs from TEST_VIDEO_SCENES scene_1."""
    return [frame["reference_id"] for frame in TEST_VIDEO_SCENES["scenes"][0]["frames"]]


@pytest.mark.integration
def test_deduplicate_video_scene_entire_dataset(dataset_video_scene):
    result = dataset_video_scene.deduplicate(threshold=10)
    assert isinstance(result, DeduplicationResult)
    assert len(result.unique_reference_ids) > 0
    assert len(result.unique_item_ids) > 0
    assert result.stats.original_count == len(_get_scene_frame_ref_ids())


@pytest.mark.integration
def test_deduplicate_video_scene_with_frame_reference_ids(dataset_video_scene):
    frame_ref_ids = _get_scene_frame_ref_ids()
    result = dataset_video_scene.deduplicate(threshold=10, reference_ids=frame_ref_ids)
    assert isinstance(result, DeduplicationResult)
    assert result.stats.original_count == len(frame_ref_ids)
    assert len(result.unique_reference_ids) <= len(frame_ref_ids)
    assert len(result.unique_item_ids) <= len(frame_ref_ids)


@pytest.mark.integration
def test_deduplicate_video_scene_by_ids(dataset_video_scene):
    initial_result = dataset_video_scene.deduplicate(threshold=10)
    item_ids = initial_result.unique_item_ids
    assert len(item_ids) > 0

    result = dataset_video_scene.deduplicate_by_ids(
        threshold=10, dataset_item_ids=item_ids
    )
    assert isinstance(result, DeduplicationResult)
    assert result.stats.original_count == len(item_ids)
    assert result.unique_item_ids == initial_result.unique_item_ids


# Edge case tests


@pytest.mark.integration
def test_deduplicate_threshold_zero(dataset_image):
    """Threshold=0 means exact matches only."""
    result = dataset_image.deduplicate(threshold=0)
    assert isinstance(result, DeduplicationResult)
    assert result.stats.threshold == 0


@pytest.mark.integration
def test_deduplicate_threshold_max(dataset_image):
    """Threshold=64 is the maximum allowed value."""
    result = dataset_image.deduplicate(threshold=64)
    assert isinstance(result, DeduplicationResult)
    assert result.stats.threshold == 64


@pytest.mark.integration
def test_deduplicate_threshold_negative(dataset_image):
    """Threshold must be >= 0."""
    with pytest.raises(NucleusAPIError):
        dataset_image.deduplicate(threshold=-1)


@pytest.mark.integration
def test_deduplicate_threshold_too_high(dataset_image):
    """Threshold must be <= 64."""
    with pytest.raises(NucleusAPIError):
        dataset_image.deduplicate(threshold=65)


@pytest.mark.integration
def test_deduplicate_threshold_non_integer(dataset_image):
    """Threshold must be an integer."""
    with pytest.raises(NucleusAPIError):
        dataset_image.deduplicate(threshold=10.5)


@pytest.mark.integration
def test_deduplicate_nonexistent_reference_id(dataset_image):
    with pytest.raises(NucleusAPIError):
        dataset_image.deduplicate(threshold=10, reference_ids=["nonexistent_ref_id"])


@pytest.mark.integration
def test_deduplicate_by_ids_nonexistent_id(dataset_image):
    with pytest.raises(NucleusAPIError):
        dataset_image.deduplicate_by_ids(threshold=10, dataset_item_ids=["di_nonexistent"])


@pytest.mark.integration
def test_deduplicate_idempotency(dataset_image):
    result1 = dataset_image.deduplicate(threshold=10)
    result2 = dataset_image.deduplicate(threshold=10)

    assert result1.unique_item_ids == result2.unique_item_ids
    assert result1.unique_reference_ids == result2.unique_reference_ids
    assert result1.stats.original_count == result2.stats.original_count
    assert result1.stats.deduplicated_count == result2.stats.deduplicated_count


@pytest.mark.integration
def test_deduplicate_response_invariants(dataset_image):
    result = dataset_image.deduplicate(threshold=10)

    assert len(result.unique_item_ids) == len(result.unique_reference_ids)
    assert result.stats.deduplicated_count == len(result.unique_item_ids)
    assert result.stats.deduplicated_count <= result.stats.original_count
    assert result.stats.threshold == 10


@pytest.mark.integration
def test_deduplicate_by_ids_threshold_negative(dataset_image):
    """deduplicate_by_ids should enforce the same threshold constraints."""
    initial_result = dataset_image.deduplicate(threshold=10)
    item_ids = initial_result.unique_item_ids

    with pytest.raises(NucleusAPIError):
        dataset_image.deduplicate_by_ids(threshold=-1, dataset_item_ids=item_ids)


@pytest.mark.integration
def test_deduplicate_by_ids_threshold_too_high(dataset_image):
    """deduplicate_by_ids should enforce the same threshold constraints."""
    initial_result = dataset_image.deduplicate(threshold=10)
    item_ids = initial_result.unique_item_ids

    with pytest.raises(NucleusAPIError):
        dataset_image.deduplicate_by_ids(threshold=65, dataset_item_ids=item_ids)


@pytest.mark.integration
def test_deduplicate_single_item(dataset_image):
    """Single item should always be unique."""
    reference_ids = [TEST_DATASET_ITEMS[0].reference_id]
    result = dataset_image.deduplicate(threshold=10, reference_ids=reference_ids)

    assert result.stats.original_count == 1
    assert result.stats.deduplicated_count == 1
    assert len(result.unique_reference_ids) == 1


@pytest.fixture()
def dataset_empty(CLIENT):
    """Empty dataset with no items."""
    ds = CLIENT.create_dataset(TEST_DATASET_NAME + " empty", is_scene=False)
    yield ds
    CLIENT.delete_dataset(ds.id)


@pytest.mark.integration
def test_deduplicate_empty_dataset(dataset_empty):
    """Empty dataset should return zero counts."""
    result = dataset_empty.deduplicate(threshold=10)

    assert result.stats.original_count == 0
    assert result.stats.deduplicated_count == 0
    assert len(result.unique_reference_ids) == 0
    assert len(result.unique_item_ids) == 0


@pytest.fixture()
def dataset_with_duplicates(CLIENT):
    """Dataset with duplicate images (same image uploaded twice)."""
    ds = CLIENT.create_dataset(TEST_DATASET_NAME + " duplicates", is_scene=False)
    items = [
        DatasetItem(image_url=TEST_IMG_URLS[0], reference_id="img_original"),
        DatasetItem(image_url=TEST_IMG_URLS[0], reference_id="img_duplicate"),
        DatasetItem(image_url=TEST_IMG_URLS[1], reference_id="img_different"),
    ]
    ds.append(items)
    yield ds
    CLIENT.delete_dataset(ds.id)


@pytest.mark.integration
def test_deduplicate_identifies_duplicates(dataset_with_duplicates):
    """Verify deduplication actually identifies duplicate images."""
    result = dataset_with_duplicates.deduplicate(threshold=0)

    assert result.stats.original_count == 3
    # With threshold=0, the two identical images should be deduplicated to one
    assert result.stats.deduplicated_count == 2
    assert len(result.unique_reference_ids) == 2


@pytest.mark.integration
def test_deduplicate_distinct_images_all_unique(dataset_image):
    """Distinct images should all remain after deduplication."""
    result = dataset_image.deduplicate(threshold=0)

    # With threshold=0 (exact match only), all distinct images should be unique
    assert result.stats.deduplicated_count == result.stats.original_count
