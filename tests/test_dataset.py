from nucleus.annotation import (
    BoxAnnotation,
    PolygonAnnotation,
    SegmentationAnnotation,
)
from nucleus.job import AsyncJob, JobError
import pytest
import os

from .helpers import (
    TEST_BOX_ANNOTATIONS,
    TEST_POLYGON_ANNOTATIONS,
    TEST_SEGMENTATION_ANNOTATIONS,
    TEST_SLICE_NAME,
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    LOCAL_FILENAME,
    reference_id_from_url,
)

from nucleus import (
    Dataset,
    DatasetItem,
    UploadResponse,
    NucleusClient,
    NucleusAPIError,
)
from nucleus.constants import (
    NEW_ITEMS,
    UPDATED_ITEMS,
    IGNORED_ITEMS,
    ERROR_ITEMS,
    ERROR_PAYLOAD,
    DATASET_ID_KEY,
)

TEST_AUTOTAG_DATASET = "ds_bz43jm2jwm70060b3890"


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    test_repr(
        DatasetItem(
            image_location="test_url",
            reference_id="test_reference_id",
            metadata={
                "made_with_pytest": True,
                "example_int": 0,
                "example_str": "hello",
                "example_float": 0.5,
                "example_dict": {
                    "nested": True,
                },
                "example_list": ["hello", 1, False],
            },
        )
    )
    test_repr(Dataset("test_dataset", NucleusClient(api_key="fake_key")))


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


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


def test_dataset_create_and_delete(CLIENT):
    # Creation
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    assert isinstance(ds, Dataset)
    assert ds.name == TEST_DATASET_NAME
    assert ds.model_runs == []
    assert ds.slices == []
    assert ds.size == 0
    assert ds.items == []

    # Deletion
    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


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
    for url in TEST_IMG_URLS:
        ds_items_plain.append(DatasetItem(image_location=url))
    response = dataset.append(ds_items_plain)
    check_is_expected_response(response)

    # With reference ids and metadata:

    response = dataset.append(make_dataset_items())
    check_is_expected_response(response)


def test_dataset_append_local(CLIENT, dataset):
    ds_items_local = [DatasetItem(image_location=LOCAL_FILENAME)]
    response = dataset.append(ds_items_local)
    assert isinstance(response, UploadResponse)
    resp_json = response.json()
    assert resp_json[DATASET_ID_KEY] == dataset.id
    assert resp_json[NEW_ITEMS] == 1
    assert resp_json[UPDATED_ITEMS] == 0
    assert resp_json[IGNORED_ITEMS] == 0
    assert resp_json[ERROR_ITEMS] == 0
    assert ERROR_PAYLOAD not in resp_json


def test_dataset_append_async(dataset: Dataset):
    job = dataset.append(make_dataset_items(), asynchronous=True)
    job.sleep_until_complete()
    status = job.status()
    assert status == {
        "job_id": job.id,
        "status": "Completed",
        "message": {
            "image_upload_step": {"errored": 0, "pending": 0, "completed": 5},
            "started_image_processing": f"Dataset: {dataset.id}, Job: {job.id}",
            "ingest_to_reupload_queue": {
                "epoch": 1,
                "total": 5,
                "datasetId": f"{dataset.id}",
                "processed": 5,
            },
        },
    }


def test_dataset_append_async_with_local_path(dataset: Dataset):
    ds_items = make_dataset_items()
    ds_items[
        0
    ].image_location = "/a/fake/local/path/you/can/tell/is/local/but/is/fake"
    with pytest.raises(ValueError):
        dataset.append(ds_items, asynchronous=True)


def test_dataset_append_async_with_1_bad_url(dataset: Dataset):
    ds_items = make_dataset_items()
    ds_items[0].image_location = "https://looks.ok.but.is.not.accessible"
    job = dataset.append(ds_items, asynchronous=True)
    with pytest.raises(JobError):
        job.sleep_until_complete()
    assert job.status() == {
        "job_id": f"{job.id}",
        "status": "Errored",
        "message": {
            "final_error": (
                "One or more of the images you attempted to upload did not process"
                " correctly. Please see the status for an overview and the errors for "
                "more detailed messages."
            ),
            "image_upload_step": {"errored": 1, "pending": 0, "completed": 4},
            "ingest_to_reupload_queue": {
                "epoch": 1,
                "total": 5,
                "datasetId": f"{dataset.id}",
                "processed": 5,
            },
            "started_image_processing": f"Dataset: {dataset.id}, Job: {job.id}",
        },
    }
    # The error is fairly detailed and subject to change. What's important is we surface which URLs failed.
    assert (
        'Failure when processing the image "https://looks.ok.but.is.not.accessible"'
        in str(job.errors())
    )


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
        str(error.value)
        == "Duplicate reference ids found among dataset_items:"
        " {'duplicate': 'Count: 2'}"
    )


def test_dataset_export_autotag_scores(CLIENT):
    # This test can only run for the test user who has an indexed dataset.
    # TODO: if/when we can create autotags via api, create one instead.
    if os.environ.get("HAS_ACCESS_TO_TEST_DATA", False):
        dataset = CLIENT.get_dataset(TEST_AUTOTAG_DATASET)

        with pytest.raises(NucleusAPIError) as api_error:
            dataset.autotag_scores(autotag_name="NONSENSE_GARBAGE")
        assert (
            f"The autotag NONSENSE_GARBAGE was not found in dataset {TEST_AUTOTAG_DATASET}"
            in str(api_error.value)
        )

        scores = dataset.autotag_scores(autotag_name="TestTag")

        for column in ["dataset_item_ids", "ref_ids", "scores"]:
            assert column in scores
            assert len(scores[column]) > 0


def test_annotate_async(dataset: Dataset):
    semseg = SegmentationAnnotation.from_json(TEST_SEGMENTATION_ANNOTATIONS[0])
    polygon = PolygonAnnotation(**TEST_POLYGON_ANNOTATIONS[0])
    bbox = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    bbox.reference_id = "fake_garbage"

    job: AsyncJob = dataset.annotate(
        annotations=[semseg, polygon, bbox],
        asynchronous=True,
    )
    job.sleep_until_complete()

    assert job.status() == {
        "job_id": job.id,
        "status": "Completed",
        "message": {
            "annotation_upload": {
                "epoch": 1,
                "total": 2,
                "errored": 0,
                "ignored": 0,
                "datasetId": dataset.id,
                "processed": 2,
            },
            "segmentation_upload": {
                "errors": [],
                "ignored": 0,
                "n_errors": 0,
                "processed": 1,
            },
        },
    }


def test_annotate_async_with_error(dataset: Dataset):
    semseg = SegmentationAnnotation.from_json(TEST_SEGMENTATION_ANNOTATIONS[0])
    polygon = PolygonAnnotation(**TEST_POLYGON_ANNOTATIONS[0])
    bbox = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    bbox.reference_id = "fake_garbage"

    job: AsyncJob = dataset.annotate(
        annotations=[semseg, polygon, bbox],
        asynchronous=True,
    )
    job.sleep_until_complete()

    assert job.status() == {
        "job_id": job.id,
        "status": "Completed",
        "message": {
            "annotation_upload": {
                "epoch": 1,
                "total": 2,
                "errored": 0,
                "ignored": 0,
                "datasetId": dataset.id,
                "processed": 1,
            },
            "segmentation_upload": {
                "errors": [],
                "ignored": 0,
                "n_errors": 0,
                "processed": 1,
            },
        },
    }

    assert "Item with id fake_garbage doesn" in str(job.errors())
