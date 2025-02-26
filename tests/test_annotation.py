import time

import pytest

from nucleus import (
    BoxAnnotation,
    CategoryAnnotation,
    DatasetItem,
    KeypointsAnnotation,
    LineAnnotation,
    MultiCategoryAnnotation,
    Point,
    PolygonAnnotation,
    Segment,
    SegmentationAnnotation,
)
from nucleus.annotation import SceneCategoryAnnotation
from nucleus.async_job import AsyncJob, JobError
from nucleus.constants import ERROR_PAYLOAD
from nucleus.errors import DuplicateIDError
from nucleus.scene import VideoScene

from .helpers import (
    TEST_BOX_ANNOTATIONS,
    TEST_BOX_ANNOTATIONS_EMBEDDINGS,
    TEST_CATEGORY_ANNOTATIONS,
    TEST_DATASET_NAME,
    TEST_DEFAULT_CATEGORY_ANNOTATIONS,
    TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS,
    TEST_IMG_URLS,
    TEST_KEYPOINTS_ANNOTATIONS,
    TEST_LINE_ANNOTATIONS,
    TEST_MULTICATEGORY_ANNOTATIONS,
    TEST_NONEXISTENT_TAXONOMY_CATEGORY_ANNOTATION,
    TEST_NONEXISTENT_TAXONOMY_SCENE_CATEGORY_ANNOTATIONS,
    TEST_POLYGON_ANNOTATIONS,
    TEST_SCENE_CATEGORY_ANNOTATIONS,
    TEST_SCENE_CATEGORY_TAXONOMY_PAYLOAD,
    TEST_SEGMENTATION_ANNOTATIONS,
    TEST_VIDEO_DATASET_NAME,
    TEST_VIDEO_SCENES_FOR_ANNOTATION,
    assert_box_annotation_matches_dict,
    assert_category_annotation_matches_dict,
    assert_keypoints_annotation_matches_dict,
    assert_line_annotation_matches_dict,
    assert_multicategory_annotation_matches_dict,
    assert_partial_equality,
    assert_polygon_annotation_matches_dict,
    reference_id_from_url,
)


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    [
        test_repr(SegmentationAnnotation.from_json(_))
        for _ in TEST_SEGMENTATION_ANNOTATIONS
    ]

    [test_repr(BoxAnnotation.from_json(_)) for _ in TEST_BOX_ANNOTATIONS]

    [
        test_repr(PolygonAnnotation.from_json(_))
        for _ in TEST_POLYGON_ANNOTATIONS
    ]


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
            )
        )

    response = ds.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()

    response = ds.add_taxonomy(
        "[Pytest] Category Taxonomy 1",
        "category",
        [f"[Pytest] Category Label ${i}" for i in range((len(TEST_IMG_URLS)))],
    )

    response = ds.add_taxonomy(
        "[Pytest] MultiCategory Taxonomy 1",
        "multicategory",
        [
            f"[Pytest] MultiCategory Label ${i}"
            for i in range((len(TEST_IMG_URLS) + 1))
        ],
    )

    yield ds


@pytest.fixture()
def scene_category_dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_VIDEO_DATASET_NAME, is_scene=True)
    scenes = []
    for scene in TEST_VIDEO_SCENES_FOR_ANNOTATION["scenes"]:
        scenes.append(VideoScene.from_json(scene, CLIENT))

    job = ds.append(
        scenes,
        asynchronous=True,
        update=TEST_VIDEO_SCENES_FOR_ANNOTATION["update"],
    )
    job.sleep_until_complete()

    ds.add_taxonomy(*TEST_SCENE_CATEGORY_TAXONOMY_PAYLOAD)

    yield ds


def test_box_gt_upload(dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"]["box"]
    single_annotation_response = dataset.ground_truth_loc(
        annotation.reference_id, annotation.annotation_id
    )

    assert response[0] == single_annotation_response
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(
        response_annotation, TEST_BOX_ANNOTATIONS[0]
    )


@pytest.mark.skip(
    reason="Skip Temporarily - Need to find issue with customObjectIndexingJobId"
)
def test_box_gt_upload_embedding(CLIENT, dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS_EMBEDDINGS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    assert response["customObjectIndexingJobId"]
    job = CLIENT.get_job(response["customObjectIndexingJobId"])
    assert job.job_last_known_status

    response = dataset.refloc(annotation.reference_id)["annotations"]["box"]
    single_annotation_response = dataset.ground_truth_loc(
        annotation.reference_id, annotation.annotation_id
    )

    assert response[0] == single_annotation_response
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(
        response_annotation, TEST_BOX_ANNOTATIONS[0]
    )


def test_line_gt_upload(dataset):
    annotation = LineAnnotation.from_json(TEST_LINE_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"]["line"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_line_annotation_matches_dict(
        response_annotation, TEST_LINE_ANNOTATIONS[0]
    )


def test_polygon_gt_upload(dataset):
    annotation = PolygonAnnotation.from_json(TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"]["polygon"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(
        response_annotation, TEST_POLYGON_ANNOTATIONS[0]
    )


def test_keypoints_gt_upload(dataset):
    annotation = KeypointsAnnotation.from_json(TEST_KEYPOINTS_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "keypoints"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_keypoints_annotation_matches_dict(
        response_annotation, TEST_KEYPOINTS_ANNOTATIONS[0]
    )


def test_category_gt_upload(dataset):
    annotation = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_category_annotation_matches_dict(
        response_annotation, TEST_CATEGORY_ANNOTATIONS[0]
    )


def test_default_category_gt_upload(dataset):
    annotation = CategoryAnnotation.from_json(
        TEST_DEFAULT_CATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]

    assert_category_annotation_matches_dict(
        response_annotation, TEST_DEFAULT_CATEGORY_ANNOTATIONS[0]
    )


@pytest.mark.skip(
    reason="Triggers SFN failed warning - need to find better unhappy tests"
)
def test_non_existent_taxonomy_category_gt_upload(dataset):
    annotation = CategoryAnnotation.from_json(
        TEST_NONEXISTENT_TAXONOMY_CATEGORY_ANNOTATION[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 0
    assert (
        f'Input validation failed: Taxonomy {TEST_NONEXISTENT_TAXONOMY_CATEGORY_ANNOTATION[0]["taxonomy_name"]} does not exist in dataset {dataset.id}'
        in response["errors"][0]
    )


def test_multicategory_gt_upload(dataset):
    annotation = MultiCategoryAnnotation.from_json(
        TEST_MULTICATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "multicategory"
    ]

    assert len(response) == 1
    response_annotation = response[0]
    assert_multicategory_annotation_matches_dict(
        response_annotation, TEST_MULTICATEGORY_ANNOTATIONS[0]
    )


def test_default_multicategory_gt_upload(dataset):
    annotation = MultiCategoryAnnotation.from_json(
        TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "multicategory"
    ]

    assert len(response) == 1
    response_annotation = response[0]
    assert_multicategory_annotation_matches_dict(
        response_annotation, TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[0]
    )


def test_scene_category_gt_upload(scene_category_dataset):
    annotation = SceneCategoryAnnotation.from_json(
        TEST_SCENE_CATEGORY_ANNOTATIONS[0]
    )
    response = scene_category_dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == scene_category_dataset.id
    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0


@pytest.mark.skip(
    reason="Triggers SFN failed warning - need to find better unhappy tests"
)
def test_non_existent_taxonomy_scene_category_gt_upload(
    scene_category_dataset,
):
    annotation = SceneCategoryAnnotation.from_json(
        TEST_NONEXISTENT_TAXONOMY_SCENE_CATEGORY_ANNOTATIONS[0]
    )
    response = scene_category_dataset.annotate(annotations=[annotation])

    assert response["dataset_id"] == scene_category_dataset.id
    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 0
    assert len(response["errors"]) > 0


def test_mixed_annotation_upload(dataset):
    # First upload annotations
    semseg_annotations = [
        SegmentationAnnotation.from_json(ann)
        for ann in TEST_SEGMENTATION_ANNOTATIONS
    ]
    bbox_annotations = [BoxAnnotation(**(ann)) for ann in TEST_BOX_ANNOTATIONS]
    annotations = bbox_annotations + semseg_annotations
    response = dataset.annotate(annotations=annotations)
    assert response["dataset_id"] == dataset.id
    assert response["annotations_processed"] == 10
    assert response["annotations_ignored"] == 0
    response_annotations = dataset.refloc(bbox_annotations[0].reference_id)[
        "annotations"
    ]

    assert len(response_annotations) == 2
    assert len(response_annotations["box"]) == 1
    assert "segmentation" in response_annotations


def test_box_gt_upload_update(dataset):
    TEST_BOX_ANNOTATIONS[0]["_task_id"] = "test_task_id"
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_BOX_ANNOTATIONS[1])
    annotation_update_params["annotation_id"] = TEST_BOX_ANNOTATIONS[0][
        "annotation_id"
    ]
    annotation_update_params["reference_id"] = TEST_BOX_ANNOTATIONS[0][
        "reference_id"
    ]
    annotation_update_params["_task_id"] = TEST_BOX_ANNOTATIONS[0]["_task_id"]

    annotation_update = BoxAnnotation(**annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"]["box"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_box_gt_upload_ignore(dataset):
    TEST_BOX_ANNOTATIONS[0]["_task_id"] = "test_task_id"
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])

    print(annotation)

    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_BOX_ANNOTATIONS[1])
    annotation_update_params["annotation_id"] = TEST_BOX_ANNOTATIONS[0][
        "annotation_id"
    ]
    annotation_update_params["reference_id"] = TEST_BOX_ANNOTATIONS[0][
        "reference_id"
    ]
    annotation_update_params["_task_id"] = TEST_BOX_ANNOTATIONS[0]["_task_id"]
    annotation_update = BoxAnnotation(**annotation_update_params)

    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"]["box"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_box_annotation_matches_dict(
        response_annotation, TEST_BOX_ANNOTATIONS[0]
    )


def test_polygon_gt_upload_update(dataset):
    TEST_POLYGON_ANNOTATIONS[0]["_task_id"] = "test_task_id"
    annotation = PolygonAnnotation.from_json(TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_POLYGON_ANNOTATIONS[1])
    annotation_update_params["annotation_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "annotation_id"
    ]
    annotation_update_params["reference_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "reference_id"
    ]
    annotation_update_params["_task_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "_task_id"
    ]

    annotation_update = PolygonAnnotation.from_json(annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"]["polygon"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_polygon_gt_upload_ignore(dataset):
    TEST_POLYGON_ANNOTATIONS[0]["_task_id"] = "test_task_id"
    annotation = PolygonAnnotation.from_json(TEST_POLYGON_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_POLYGON_ANNOTATIONS[1])
    annotation_update_params["annotation_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "annotation_id"
    ]
    annotation_update_params["reference_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "reference_id"
    ]
    annotation_update_params["_task_id"] = TEST_POLYGON_ANNOTATIONS[0][
        "_task_id"
    ]

    annotation_update = PolygonAnnotation.from_json(annotation_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"]["polygon"]
    assert len(response) == 1
    response_annotation = response[0]
    assert_polygon_annotation_matches_dict(
        response_annotation, TEST_POLYGON_ANNOTATIONS[0]
    )


def test_category_gt_upload_update(dataset):
    annotation = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_CATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = TEST_CATEGORY_ANNOTATIONS[0][
        "reference_id"
    ]

    annotation_update = CategoryAnnotation.from_json(annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    # TODO(gunnar): Remove this sleep -> This is added due to flakiness. Might be replication lag?
    time.sleep(2)
    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_category_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_category_gt_upload_ignore(dataset):
    annotation = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_CATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = TEST_CATEGORY_ANNOTATIONS[0][
        "reference_id"
    ]

    annotation_update = CategoryAnnotation.from_json(annotation_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_category_annotation_matches_dict(
        response_annotation, TEST_CATEGORY_ANNOTATIONS[0]
    )


# TODO(drake): investigate why this only flakes in circleci
@pytest.mark.skip(reason="Flaky test")
def test_default_category_gt_upload_update(dataset):
    annotation = CategoryAnnotation.from_json(
        TEST_DEFAULT_CATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_DEFAULT_CATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = (
        TEST_DEFAULT_CATEGORY_ANNOTATIONS[0]["reference_id"]
    )

    annotation_update = CategoryAnnotation.from_json(annotation_update_params)
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_category_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_default_category_gt_upload_ignore(dataset):
    annotation = CategoryAnnotation.from_json(
        TEST_DEFAULT_CATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_DEFAULT_CATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = (
        TEST_DEFAULT_CATEGORY_ANNOTATIONS[0]["reference_id"]
    )

    annotation_update = CategoryAnnotation.from_json(annotation_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "category"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_category_annotation_matches_dict(
        response_annotation, TEST_DEFAULT_CATEGORY_ANNOTATIONS[0]
    )


def test_multicategory_gt_upload_update(dataset):
    annotation = MultiCategoryAnnotation.from_json(
        TEST_MULTICATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_MULTICATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = TEST_MULTICATEGORY_ANNOTATIONS[
        0
    ]["reference_id"]

    annotation_update = MultiCategoryAnnotation.from_json(
        annotation_update_params
    )
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "multicategory"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_multicategory_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_multicategory_gt_upload_ignore(dataset):
    annotation = MultiCategoryAnnotation.from_json(
        TEST_MULTICATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_MULTICATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = TEST_MULTICATEGORY_ANNOTATIONS[
        0
    ]["reference_id"]

    annotation_update = MultiCategoryAnnotation.from_json(
        annotation_update_params
    )
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "multicategory"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_multicategory_annotation_matches_dict(
        response_annotation, TEST_MULTICATEGORY_ANNOTATIONS[0]
    )


def test_default_multicategory_gt_upload_update(dataset):
    annotation = MultiCategoryAnnotation.from_json(
        TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = (
        TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[0]["reference_id"]
    )

    annotation_update = MultiCategoryAnnotation.from_json(
        annotation_update_params
    )
    response = dataset.annotate(annotations=[annotation_update], update=True)

    assert response["annotations_processed"] == 1
    assert response["annotations_ignored"] == 0

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "multicategory"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_multicategory_annotation_matches_dict(
        response_annotation, annotation_update_params
    )


def test_default_multicategory_gt_upload_ignore(dataset):
    annotation = MultiCategoryAnnotation.from_json(
        TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[0]
    )
    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    # Copy so we don't modify the original.
    annotation_update_params = dict(TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[1])
    annotation_update_params["reference_id"] = (
        TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[0]["reference_id"]
    )

    annotation_update = MultiCategoryAnnotation.from_json(
        annotation_update_params
    )
    # Default behavior is ignore.
    response = dataset.annotate(annotations=[annotation_update])

    assert response["annotations_processed"] == 0
    assert response["annotations_ignored"] == 1

    response = dataset.refloc(annotation.reference_id)["annotations"][
        "multicategory"
    ]
    assert len(response) == 1
    response_annotation = response[0]
    assert_multicategory_annotation_matches_dict(
        response_annotation, TEST_DEFAULT_MULTICATEGORY_ANNOTATIONS[0]
    )


@pytest.mark.integration
def test_box_gt_deletion(dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])

    print(annotation)

    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    job = dataset.delete_annotations(
        reference_ids=[TEST_BOX_ANNOTATIONS[0]["reference_id"]]
    )
    job.sleep_until_complete()
    job_status = job.status()
    assert job_status["status"] == "Completed"
    assert job_status["job_id"] == job.job_id


@pytest.mark.integration
def test_category_gt_deletion(dataset):
    annotation = CategoryAnnotation.from_json(TEST_CATEGORY_ANNOTATIONS[0])

    print(annotation)

    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    job = dataset.delete_annotations(
        reference_ids=[TEST_CATEGORY_ANNOTATIONS[0]["reference_id"]]
    )
    job.sleep_until_complete()
    job_status = job.status()
    assert job_status["status"] == "Completed"
    assert job_status["job_id"] == job.job_id


@pytest.mark.integration
def test_multicategory_gt_deletion(dataset):
    annotation = MultiCategoryAnnotation.from_json(
        TEST_MULTICATEGORY_ANNOTATIONS[0]
    )

    print(annotation)

    response = dataset.annotate(annotations=[annotation])

    assert response["annotations_processed"] == 1

    job = dataset.delete_annotations(
        reference_ids=[TEST_MULTICATEGORY_ANNOTATIONS[0]["reference_id"]]
    )
    job.sleep_until_complete()
    job_status = job.status()
    assert job_status["status"] == "Completed"
    assert job_status["job_id"] == job.job_id


@pytest.mark.integration
def test_default_category_gt_upload_async(dataset):
    annotation = CategoryAnnotation.from_json(
        TEST_DEFAULT_CATEGORY_ANNOTATIONS[0]
    )
    job: AsyncJob = dataset.annotate(
        annotations=[
            annotation,
        ],
        asynchronous=True,
    )
    job.sleep_until_complete()
    result = job.status()

    expected = {
        "job_id": job.job_id,
        "status": "Completed",
        "message": {
            "annotation_upload": {
                "total": 1,
                "errored": 0,
                "ignored": 0,
                "datasetId": dataset.id,
                "processed": 1,
            },
        },
        "job_progress": "1.00",
        "completed_steps": 1,
        "total_steps": 1,
    }
    assert_partial_equality(expected, result)


@pytest.mark.skip(
    reason="Triggers SFN failed warning - need to find better unhappy tests"
)
@pytest.mark.integration
def test_non_existent_taxonomy_category_gt_upload_async(dataset):
    annotation = CategoryAnnotation.from_json(
        TEST_NONEXISTENT_TAXONOMY_CATEGORY_ANNOTATION[0]
    )
    error_msg = f'Input validation failed: Taxonomy {TEST_NONEXISTENT_TAXONOMY_CATEGORY_ANNOTATION[0]["taxonomy_name"]} does not exist in dataset {dataset.id}, or label {annotation.label} does not exist in the taxonomy {TEST_NONEXISTENT_TAXONOMY_CATEGORY_ANNOTATION[0]["taxonomy_name"]}.'

    try:
        job: AsyncJob = dataset.annotate(
            annotations=[
                annotation,
            ],
            asynchronous=True,
        )
        job.sleep_until_complete()
    except JobError:
        assert error_msg in job.errors()[-1]

    result = job.status()

    expected = {
        "job_id": job.job_id,
        "status": "Errored",
        "message": {
            "final_error": f"BadRequestError: {error_msg}",
        },
        "job_progress": "1.00",
        "completed_steps": 1,
        "total_steps": 1,
    }

    assert_partial_equality(expected, result)


@pytest.mark.skip(
    reason="Skip Temporarily - Need to find issue with customObjectIndexingJobId"
)
@pytest.mark.integration
def test_box_gt_upload_embedding_async(CLIENT, dataset):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS_EMBEDDINGS[0])
    job = dataset.annotate(annotations=[annotation], asynchronous=True)
    status = job.status()
    assert status["job_id"] == job.job_id
    assert status["status"] == "Running"


def test_annotation_duplicate_ids_fail(dataset):
    box_ann = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    annotations = [box_ann, box_ann]
    with pytest.raises(DuplicateIDError):
        dataset.annotate(annotations=annotations)
