import copy
import glob
import math
import os

import pytest

from nucleus import Dataset, DatasetItem, NucleusClient, UploadResponse
from nucleus.annotation import (
    BoxAnnotation,
    CategoryAnnotation,
    MultiCategoryAnnotation,
    PolygonAnnotation,
    SegmentationAnnotation,
)
from nucleus.async_job import AsyncJob, JobError
from nucleus.constants import (
    ANNOTATIONS_KEY,
    BOX_TYPE,
    CATEGORY_TYPE,
    DATASET_ID_KEY,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    IGNORED_ITEMS,
    ITEM_KEY,
    MULTICATEGORY_TYPE,
    NEW_ITEMS,
    POLYGON_TYPE,
    SEGMENTATION_TYPE,
    UPDATED_ITEMS,
)
from nucleus.errors import NucleusAPIError
from nucleus.scene import LidarScene, VideoScene

from .helpers import (
    DATASET_WITH_EMBEDDINGS,
    LOCAL_FILENAME,
    TEST_BOX_ANNOTATIONS,
    TEST_CATEGORY_ANNOTATIONS,
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    TEST_LIDAR_SCENES,
    TEST_LOCAL_TESTDIR,
    TEST_MULTICATEGORY_ANNOTATIONS,
    TEST_POLYGON_ANNOTATIONS,
    TEST_SEGMENTATION_ANNOTATIONS,
    TEST_VIDEO_SCENES,
    assert_partial_equality,
    reference_id_from_url,
)


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME, is_scene=False)

    ds.add_taxonomy(
        "[Pytest] Category Taxonomy 1",
        "category",
        [f"[Pytest] Category Label ${i}" for i in range((len(TEST_IMG_URLS)))],
    )

    ds.add_taxonomy(
        "[Pytest] MultiCategory Taxonomy 1",
        "multicategory",
        [
            f"[Pytest] MultiCategory Label ${i}"
            for i in range((len(TEST_IMG_URLS) + 1))
        ],
    )

    yield ds


@pytest.fixture()
def dataset_scene(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME, is_scene=True)
    yield ds


def make_dataset_items():
    ds_items_with_metadata = []
    for i, url in enumerate(TEST_IMG_URLS):
        ds_items_with_metadata.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
                metadata={
                    "made_with_pytest": True,
                    "example_int": i,
                    "example_str": "hello",
                    "example_float": 0.5,
                    "example_dict": {
                        "nested": True,
                    },
                    "example_list": ["hello", i, False],
                },
            )
        )
    return ds_items_with_metadata


def make_scenes():
    return [VideoScene.from_json(s) for s in TEST_VIDEO_SCENES["scenes"]]


def test_dataset_create_and_delete_no_scene(CLIENT):
    # Creation
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    assert isinstance(ds, Dataset)
    assert ds.name == TEST_DATASET_NAME
    assert not ds.is_scene
    assert ds.model_runs == []
    assert ds.slices == []
    assert ds.size == 0
    assert ds.items == []

    # Deletion
    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


def test_dataset_create_and_delete_scene(CLIENT):
    # Creation
    ds = CLIENT.create_dataset(name=TEST_DATASET_NAME, is_scene=True)
    assert isinstance(ds, Dataset)
    assert ds.name == TEST_DATASET_NAME
    assert ds.is_scene
    assert ds.model_runs == []
    assert ds.slices == []
    assert ds.size == 0
    assert ds.items == []

    # Deletion
    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


def test_dataset_update_metadata_local(dataset):
    dataset.append(
        [
            DatasetItem(
                image_location=LOCAL_FILENAME,
                metadata={"snake_field": 0},
                reference_id="test_image",
            )
        ]
    )
    dataset.append(
        [
            DatasetItem(
                image_location=LOCAL_FILENAME,
                metadata={"snake_field": 1},
                reference_id="test_image",
            )
        ],
        update=True,
    )
    resulting_item = dataset.iloc(0)["item"]
    print(resulting_item)
    assert resulting_item.metadata["snake_field"] == 1


def test_dataset_update_metadata(dataset):
    dataset.append(
        [
            DatasetItem(
                image_location=TEST_IMG_URLS[0],
                metadata={"snake_field": 0},
                reference_id="test_image",
            )
        ]
    )
    dataset.append(
        [
            DatasetItem(
                image_location=TEST_IMG_URLS[0],
                metadata={"snake_field": 1},
                reference_id="test_image",
            )
        ],
        update=True,
    )
    resulting_item = dataset.iloc(0)["item"]
    print(resulting_item)
    assert resulting_item.metadata["snake_field"] == 1


def test_dataset_append(dataset):
    def check_is_expected_response(response):
        assert isinstance(response, UploadResponse)
        resp_json = response.json()
        assert resp_json[DATASET_ID_KEY] == dataset.id
        assert resp_json[NEW_ITEMS] == len(TEST_IMG_URLS)
        assert resp_json[UPDATED_ITEMS] == 0
        assert resp_json[IGNORED_ITEMS] == 0
        assert resp_json[ERROR_ITEMS] == 0
        assert ERROR_PAYLOAD not in resp_json

    # Plain image upload
    ds_items_plain = []
    for i, url in enumerate(TEST_IMG_URLS):
        ds_items_plain.append(
            DatasetItem(
                image_location=url,
                reference_id=url.split("/")[-1] + "_plain",
            )
        )

    response = dataset.append(ds_items_plain)
    check_is_expected_response(response)

    # With reference ids and metadata:

    response = dataset.append(make_dataset_items())
    check_is_expected_response(response)


def test_scene_dataset_append(dataset_scene):
    # Plain image upload
    ds_items_plain = []
    for i, url in enumerate(TEST_IMG_URLS):
        ds_items_plain.append(
            DatasetItem(
                image_location=url,
                reference_id=url.split("/")[-1] + "_plain",
            )
        )

    with pytest.raises(Exception):
        dataset_scene.append(ds_items_plain)


def test_dataset_name_access(CLIENT, dataset):
    assert dataset.name == TEST_DATASET_NAME


def test_dataset_size_access(CLIENT, dataset):
    assert dataset.size == 0
    items = make_dataset_items()
    dataset.append(items)
    assert dataset.size == len(items)


def test_dataset_model_runs_access(CLIENT, dataset):
    # TODO: Change to Models
    assert len(dataset.model_runs) == 0


def test_dataset_slices(CLIENT, dataset):
    assert len(dataset.slices) == 0
    items = make_dataset_items()
    dataset.append(items)
    dataset.create_slice("test_slice", [item.reference_id for item in items])
    slices = dataset.slices
    assert len(slices) == 1
    # TODO(gunnar): Test slice items -> Split up info!


def test_dataset_append_local(CLIENT, dataset):
    ds_items_local_error = [
        DatasetItem(
            image_location=LOCAL_FILENAME,
            metadata={"test": math.nan},
            reference_id="bad",
        )
    ]
    num_local_items_to_test = 10
    with pytest.raises(ValueError) as e:
        dataset.append(ds_items_local_error)
        assert "Out of range float values are not JSON compliant" in str(
            e.value
        )
    ds_items_local = [
        DatasetItem(
            image_location=LOCAL_FILENAME,
            metadata={"test": 0},
            reference_id=LOCAL_FILENAME.split("/")[-1] + str(i),
        )
        for i in range(num_local_items_to_test)
    ]

    response = dataset.append(ds_items_local)

    assert isinstance(response, UploadResponse)
    resp_json = response.json()
    assert resp_json[DATASET_ID_KEY] == dataset.id
    assert resp_json[NEW_ITEMS] == num_local_items_to_test
    assert resp_json[UPDATED_ITEMS] == 0
    assert resp_json[IGNORED_ITEMS] == 0
    assert resp_json[ERROR_ITEMS] == 0
    assert ERROR_PAYLOAD not in resp_json


@pytest.mark.integration
def test_dataset_append_async(dataset: Dataset):
    job = dataset.append(make_dataset_items(), asynchronous=True)
    job.sleep_until_complete()
    status = job.status()
    expected = {
        "job_id": job.job_id,
        "status": "Completed",
        "job_progress": "1.00",
        "completed_steps": 5,
        "total_steps": 5,
    }
    assert_partial_equality(expected, status)


def test_dataset_append_async_with_local_path(dataset: Dataset):
    ds_items = make_dataset_items()
    ds_items[0].image_location = (
        "/a/fake/local/path/you/can/tell/is/local/but/is/fake"
    )
    with pytest.raises(ValueError):
        dataset.append(ds_items, asynchronous=True)


# TODO(Jean): Fix and remove skip, this is a flaky test
@pytest.mark.skip(reason="Flaky test")
def test_dataset_append_async_with_1_bad_url(dataset: Dataset):
    ds_items = make_dataset_items()
    ds_items[0].image_location = "https://looks.ok.but.is.not.accessible"
    job = dataset.append(ds_items, asynchronous=True)
    with pytest.raises(JobError):
        job.sleep_until_complete()
    status = job.status()
    status["message"]["PayloadUrl"] = ""
    print("STATUS: ")
    print(status)
    assert status["job_id"] == job.job_id
    assert status["status"] == "Errored"
    assert status["job_progress"] == "0.80"
    assert status["completed_steps"] == 4
    assert status["total_steps"] == 5
    # The error is fairly detailed and subject to change. What's important is we surface which URLs failed.
    assert '"https://looks.ok.but.is.not.accessible"' in str(job.errors())


def test_dataset_list_autotags(CLIENT, dataset):
    # Creation
    # List of Autotags should be empty
    autotag_response = CLIENT.list_autotags(dataset.id)
    assert autotag_response == []


def test_raises_error_for_duplicate():
    fake_dataset = Dataset("fake", NucleusClient("fake"))
    with pytest.raises(ValueError) as error:
        fake_dataset.append(
            [
                DatasetItem("fake", "duplicate"),
                DatasetItem("fake", "duplicate"),
            ]
        )
    assert (
        str(error.value) == "Duplicate reference IDs found among dataset_items:"
        " {'duplicate': 'Count: 2'}"
    )


@pytest.mark.integration
def test_annotate_async(dataset: Dataset):
    dataset.append(make_dataset_items())
    semseg = SegmentationAnnotation.from_json(TEST_SEGMENTATION_ANNOTATIONS[0])
    polygon = PolygonAnnotation.from_json(TEST_POLYGON_ANNOTATIONS[0])
    bbox = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    category = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])
    multicategory = MultiCategoryAnnotation.from_json(
        TEST_MULTICATEGORY_ANNOTATIONS[0]
    )

    job: AsyncJob = dataset.annotate(
        annotations=[semseg, polygon, bbox, category, multicategory],
        asynchronous=True,
    )
    job.sleep_until_complete()
    status = job.status()
    expected = {
        "job_id": job.job_id,
        "status": "Completed",
    }
    assert_partial_equality(expected, status)


@pytest.mark.integration
def test_annotate_async_with_error(dataset: Dataset):
    dataset.append(make_dataset_items())
    semseg = SegmentationAnnotation.from_json(TEST_SEGMENTATION_ANNOTATIONS[0])
    polygon = PolygonAnnotation.from_json(TEST_POLYGON_ANNOTATIONS[0])
    category = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])
    multicategory = MultiCategoryAnnotation.from_json(
        TEST_MULTICATEGORY_ANNOTATIONS[0]
    )
    bbox = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    bbox.reference_id = "fake_garbage"

    job: AsyncJob = dataset.annotate(
        annotations=[semseg, polygon, bbox, category, multicategory],
        asynchronous=True,
    )
    job.sleep_until_complete()
    status = job.status()
    expected = {
        "job_id": job.job_id,
        "status": "Completed",
    }
    assert_partial_equality(expected, status)

    assert "Item with id fake_garbage doesn" in str(job.errors())


def test_append_with_special_chars(dataset):
    url = TEST_IMG_URLS[0]
    ref_id = "test/reference/id"
    ds_items = [
        DatasetItem(
            image_location=url,
            reference_id=ref_id,
            metadata={"test": "metadata"},
        ),
    ]
    dataset.append(ds_items)
    dataset.refloc(ref_id)


def test_append_and_export(dataset):
    # Dataset upload
    url = TEST_IMG_URLS[0]
    box_annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    segmentation_annotation = SegmentationAnnotation.from_json(
        TEST_SEGMENTATION_ANNOTATIONS[0]
    )
    polygon_annotation = PolygonAnnotation.from_json(
        TEST_POLYGON_ANNOTATIONS[0]
    )
    category_annotation = CategoryAnnotation.from_json(
        TEST_CATEGORY_ANNOTATIONS[0]
    )
    multicategory_annotation = MultiCategoryAnnotation.from_json(
        TEST_MULTICATEGORY_ANNOTATIONS[0]
    )

    ds_items = [
        DatasetItem(
            image_location=url,
            reference_id=reference_id_from_url(url),
            metadata={"test": "metadata"},
        ),
    ]
    response = dataset.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()

    dataset.annotate(
        annotations=[
            box_annotation,
            polygon_annotation,
            segmentation_annotation,
            category_annotation,
            multicategory_annotation,
        ]
    )
    # We don't export everything on segmentation annotations in order to speed up export.

    def clear_fields(annotation):
        cleared_annotation = copy.deepcopy(annotation)
        cleared_annotation.metadata = {}
        return cleared_annotation

    def sort_labelmap(segmentation_annotation):
        segmentation_annotation.annotations = sorted(
            segmentation_annotation.annotations, key=lambda x: x.index
        )

    exported = dataset.items_and_annotations()
    assert exported[0][ITEM_KEY] == ds_items[0]
    assert exported[0][ANNOTATIONS_KEY][BOX_TYPE][0] == box_annotation
    assert sort_labelmap(
        exported[0][ANNOTATIONS_KEY][SEGMENTATION_TYPE]
    ) == sort_labelmap(clear_fields(segmentation_annotation))
    assert exported[0][ANNOTATIONS_KEY][POLYGON_TYPE][0] == polygon_annotation
    assert exported[0][ANNOTATIONS_KEY][CATEGORY_TYPE][0] == category_annotation
    exported[0][ANNOTATIONS_KEY][MULTICATEGORY_TYPE][0].labels = set(
        exported[0][ANNOTATIONS_KEY][MULTICATEGORY_TYPE][0].labels
    )
    multicategory_annotation.labels = set(multicategory_annotation.labels)
    assert (
        exported[0][ANNOTATIONS_KEY][MULTICATEGORY_TYPE][0]
        == multicategory_annotation
    )

    # test async export
    for row in dataset.items_and_annotation_generator():
        assert row[ITEM_KEY] == ds_items[0]
        assert row[ANNOTATIONS_KEY][BOX_TYPE][0] == box_annotation
        assert sort_labelmap(
            row[ANNOTATIONS_KEY][SEGMENTATION_TYPE]
        ) == sort_labelmap(clear_fields(segmentation_annotation))
        assert row[ANNOTATIONS_KEY][POLYGON_TYPE][0] == polygon_annotation
        assert row[ANNOTATIONS_KEY][CATEGORY_TYPE][0] == category_annotation
        row[ANNOTATIONS_KEY][MULTICATEGORY_TYPE][0].labels = set(
            row[ANNOTATIONS_KEY][MULTICATEGORY_TYPE][0].labels
        )
        multicategory_annotation.labels = set(multicategory_annotation.labels)
        assert (
            row[ANNOTATIONS_KEY][MULTICATEGORY_TYPE][0]
            == multicategory_annotation
        )


def test_dataset_item_metadata_update(dataset):
    items = make_dataset_items()
    dataset.append(items)

    expected_metadata = {}
    new_metadata = {}
    for item in dataset.items:
        data = {"a_new_key": 123}
        new_metadata[item.reference_id] = data
        expected_metadata[item.reference_id] = {**item.metadata, **data}

    dataset.update_item_metadata(new_metadata)
    actual_metadata = {
        item.reference_id: item.metadata for item in dataset.items
    }

    assert actual_metadata == expected_metadata


def test_dataset_item_iterator(dataset):
    items = make_dataset_items()
    dataset.append(items)
    expected_items = {item.reference_id: item for item in dataset.items}
    actual_items = {
        item.reference_id: item for item in dataset.items_generator(page_size=1)
    }
    for key in expected_items:
        assert actual_items[key] == expected_items[key]


@pytest.mark.integration
def test_dataset_get_image_indexing_status(CLIENT):
    dataset = Dataset(DATASET_WITH_EMBEDDINGS, CLIENT)
    resp = dataset.get_image_indexing_status()
    print(resp)
    assert resp["embedding_count"] == 170
    assert resp["image_count"] == 170
    assert "object_count" not in resp


@pytest.mark.integration
def test_dataset_get_object_indexing_status(CLIENT):
    dataset = Dataset(DATASET_WITH_EMBEDDINGS, CLIENT)
    resp = dataset.get_object_indexing_status()
    assert resp["embedding_count"] == 422
    assert resp["object_count"] == 423
    assert "image_count" not in resp


@pytest.mark.integration
def test_query(CLIENT):
    dataset = Dataset(DATASET_WITH_EMBEDDINGS, CLIENT)
    expected_items = {
        ia["item"].reference_id: ia["item"]
        for ia in dataset.items_and_annotations()
        if len(ia["annotations"]["box"]) > 6  # assume only box annotations
    }
    queried_items = [i for i in dataset.query_items("annotations.count > 6")]

    assert len(queried_items) == len(expected_items)
    for qi in queried_items:
        assert qi == expected_items[qi.reference_id]

    with pytest.raises(NucleusAPIError):
        for qi in dataset.query_items("annotations.count bad syntax"):
            print(qi)  # unreachable, just need to yield an item from generator


@pytest.mark.integration
def test_create_update_dataset_from_dir(CLIENT):
    reference_ids = set()
    for file_type in ["png", "jpeg"]:
        pathname = os.path.join(TEST_LOCAL_TESTDIR, f"**/*.{file_type}")
        reference_ids.update(
            path.replace(TEST_LOCAL_TESTDIR + "/", "")
            for path in glob.glob(pathname=pathname, recursive=True)
        )
    dataset = CLIENT.create_dataset_from_dir(
        TEST_LOCAL_TESTDIR, allowed_file_types=tuple(["exe"])
    )
    assert dataset is not None
    CLIENT.delete_dataset(dataset.id)
    dataset = CLIENT.create_dataset_from_dir(
        TEST_LOCAL_TESTDIR, allowed_file_types=tuple(["png"])
    )
    dataset_items = dataset.items
    assert len(dataset_items) == 1
    assert dataset_items[0].reference_id in reference_ids
    dataset.add_items_from_dir(
        dirname=TEST_LOCAL_TESTDIR,
        allowed_file_types=tuple(["png", "jpeg"]),
    )
    dataset_items = dataset.items
    assert len(dataset_items) == 2
    for dataset_item in dataset_items:
        assert dataset_item.reference_id in reference_ids
        reference_ids.remove(dataset_item.reference_id)
    CLIENT.delete_dataset(dataset.id)


@pytest.mark.integration
def test_dataset_export_class_labels(dataset):
    dataset.append(make_dataset_items())
    # Create box annotation from the test data
    box_annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    dataset.annotate(annotations=[box_annotation])

    # Wait annotations to be uploaded (takes a while)
    import time

    time.sleep(40)
    class_labels = dataset.export_class_labels()
    # Compare against just the label from the test annotation
    assert class_labels == [box_annotation.label]
