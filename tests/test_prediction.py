import time

import pytest

from nucleus import (
    BoxPrediction,
    CategoryPrediction,
    DatasetItem,
    KeypointsPrediction,
    LinePrediction,
    ModelRun,
    PolygonPrediction,
    SegmentationPrediction,
)
from nucleus.constants import ERROR_PAYLOAD
from nucleus.errors import DuplicateIDError
from nucleus.job import AsyncJob, JobError

from .helpers import (
    TEST_BOX_PREDICTIONS,
    TEST_BOX_PREDICTIONS_EMBEDDINGS,
    TEST_CATEGORY_PREDICTIONS,
    TEST_DATASET_NAME,
    TEST_DEFAULT_CATEGORY_PREDICTIONS,
    TEST_IMG_URLS,
    TEST_KEYPOINTS_PREDICTIONS,
    TEST_LINE_PREDICTIONS,
    TEST_MODEL_NAME,
    TEST_MODEL_RUN,
    TEST_NONEXISTENT_TAXONOMY_CATEGORY_PREDICTION,
    TEST_POLYGON_PREDICTIONS,
    TEST_SEGMENTATION_PREDICTIONS,
    assert_box_prediction_matches_dict,
    assert_category_prediction_matches_dict,
    assert_keypoints_prediction_matches_dict,
    assert_line_prediction_matches_dict,
    assert_polygon_prediction_matches_dict,
    assert_segmentation_annotation_matches_dict,
    reference_id_from_url,
)


def test_reprs():
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    [
        test_repr(SegmentationPrediction.from_json(_))
        for _ in TEST_SEGMENTATION_PREDICTIONS
    ]

    [test_repr(BoxPrediction.from_json(_)) for _ in TEST_BOX_PREDICTIONS]

    [test_repr(LinePrediction.from_json(_)) for _ in TEST_LINE_PREDICTIONS]

    [
        test_repr(PolygonPrediction.from_json(_))
        for _ in TEST_POLYGON_PREDICTIONS
    ]

    [
        test_repr(CategoryPrediction.from_json(_))
        for _ in TEST_CATEGORY_PREDICTIONS
    ]

    [
        test_repr(CategoryPrediction.from_json(_))
        for _ in TEST_DEFAULT_CATEGORY_PREDICTIONS
    ]


@pytest.fixture()
def model_run(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds_items = []
    for url in TEST_IMG_URLS[:2]:
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

    model = CLIENT.create_model(
        name=TEST_MODEL_NAME, reference_id="model_" + str(time.time())
    )

    run = model.create_run(name=TEST_MODEL_RUN, dataset=ds, predictions=[])

    yield run

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}
    response = CLIENT.delete_model(model.id)
    assert response == {}


def test_box_pred_upload(model_run):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["box"]
    single_prediction = model_run.prediction_loc(
        prediction.reference_id, prediction.annotation_id
    )
    assert response[0] == single_prediction
    assert len(response) == 1
    assert_box_prediction_matches_dict(response[0], TEST_BOX_PREDICTIONS[0])


def test_box_pred_upload_embedding(CLIENT, model_run):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS_EMBEDDINGS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    assert response["customObjectIndexingJobId"]
    job = CLIENT.get_job(response["customObjectIndexingJobId"])
    assert job.job_last_known_status

    response = model_run.refloc(prediction.reference_id)["box"]
    single_prediction = model_run.prediction_loc(
        prediction.reference_id, prediction.annotation_id
    )
    assert response[0] == single_prediction
    assert len(response) == 1
    assert_box_prediction_matches_dict(response[0], TEST_BOX_PREDICTIONS[0])


def test_line_pred_upload(model_run):
    prediction = LinePrediction.from_json(TEST_LINE_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["line"]
    assert len(response) == 1
    assert_line_prediction_matches_dict(response[0], TEST_LINE_PREDICTIONS[0])


def test_polygon_pred_upload(model_run):
    prediction = PolygonPrediction.from_json(TEST_POLYGON_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["polygon"]
    assert len(response) == 1
    assert_polygon_prediction_matches_dict(
        response[0], TEST_POLYGON_PREDICTIONS[0]
    )


def test_keypoints_pred_upload(model_run):
    prediction = KeypointsPrediction.from_json(TEST_KEYPOINTS_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["keypoints"]
    assert len(response) == 1
    assert_keypoints_prediction_matches_dict(
        response[0], TEST_KEYPOINTS_PREDICTIONS[0]
    )


def test_category_pred_upload(model_run):
    prediction = CategoryPrediction.from_json(TEST_CATEGORY_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["category"]
    assert len(response) == 1
    assert_category_prediction_matches_dict(
        response[0], TEST_CATEGORY_PREDICTIONS[0]
    )


def test_default_category_pred_upload(model_run):
    prediction = CategoryPrediction.from_json(
        TEST_DEFAULT_CATEGORY_PREDICTIONS[0]
    )
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["category"]
    assert len(response) == 1
    assert_category_prediction_matches_dict(
        response[0], TEST_DEFAULT_CATEGORY_PREDICTIONS[0]
    )


@pytest.mark.skip("Need to adjust error message on taxonomy failure")
def test_non_existent_taxonomy_category_gt_upload(model_run):
    prediction = CategoryPrediction.from_json(
        TEST_NONEXISTENT_TAXONOMY_CATEGORY_PREDICTION[0]
    )
    response = model_run.predict(annotations=[prediction])
    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 0
    assert (
        f'Input validation failed: Taxonomy {TEST_NONEXISTENT_TAXONOMY_CATEGORY_PREDICTION[0]["taxonomy_name"]} does not exist in dataset'
        in response["errors"][0]
    )


def test_box_pred_upload_update(model_run):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_BOX_PREDICTIONS[1])
    prediction_update_params["annotation_id"] = TEST_BOX_PREDICTIONS[0][
        "annotation_id"
    ]
    prediction_update_params["reference_id"] = TEST_BOX_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = BoxPrediction(**prediction_update_params)
    response = model_run.predict(annotations=[prediction_update], update=True)

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["box"]
    assert len(response) == 1
    assert_box_prediction_matches_dict(response[0], prediction_update_params)


def test_box_pred_upload_ignore(model_run):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_BOX_PREDICTIONS[1])
    prediction_update_params["annotation_id"] = TEST_BOX_PREDICTIONS[0][
        "annotation_id"
    ]
    prediction_update_params["reference_id"] = TEST_BOX_PREDICTIONS[0][
        "reference_id"
    ]
    prediction_update = BoxPrediction(**prediction_update_params)
    # Default behavior is ignore.
    response = model_run.predict(annotations=[prediction_update])

    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)["box"]
    assert len(response) == 1
    assert_box_prediction_matches_dict(response[0], TEST_BOX_PREDICTIONS[0])


def test_line_pred_upload_update(model_run):
    prediction = LinePrediction.from_json(TEST_LINE_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_LINE_PREDICTIONS[1])
    prediction_update_params["annotation_id"] = TEST_LINE_PREDICTIONS[0][
        "annotation_id"
    ]
    prediction_update_params["reference_id"] = TEST_LINE_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = LinePrediction.from_json(prediction_update_params)
    response = model_run.predict(annotations=[prediction_update], update=True)

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["line"]
    assert len(response) == 1
    assert_line_prediction_matches_dict(response[0], prediction_update_params)


def test_line_pred_upload_ignore(model_run):
    prediction = LinePrediction.from_json(TEST_LINE_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_LINE_PREDICTIONS[1])
    prediction_update_params["annotation_id"] = TEST_LINE_PREDICTIONS[0][
        "annotation_id"
    ]
    prediction_update_params["reference_id"] = TEST_LINE_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = LinePrediction.from_json(prediction_update_params)
    # Default behavior is ignore.
    response = model_run.predict(annotations=[prediction_update])

    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)["line"]
    assert len(response) == 1
    assert_line_prediction_matches_dict(response[0], TEST_LINE_PREDICTIONS[0])


def test_polygon_pred_upload_update(model_run):
    prediction = PolygonPrediction.from_json(TEST_POLYGON_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_POLYGON_PREDICTIONS[1])
    prediction_update_params["annotation_id"] = TEST_POLYGON_PREDICTIONS[0][
        "annotation_id"
    ]
    prediction_update_params["reference_id"] = TEST_POLYGON_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = PolygonPrediction.from_json(prediction_update_params)
    response = model_run.predict(annotations=[prediction_update], update=True)

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["polygon"]
    assert len(response) == 1
    assert_polygon_prediction_matches_dict(
        response[0], prediction_update_params
    )


def test_polygon_pred_upload_ignore(model_run):
    prediction = PolygonPrediction.from_json(TEST_POLYGON_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_POLYGON_PREDICTIONS[1])
    prediction_update_params["annotation_id"] = TEST_POLYGON_PREDICTIONS[0][
        "annotation_id"
    ]
    prediction_update_params["reference_id"] = TEST_POLYGON_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = PolygonPrediction.from_json(prediction_update_params)
    # Default behavior is ignore.
    response = model_run.predict(annotations=[prediction_update])

    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)["polygon"]
    assert len(response) == 1
    assert_polygon_prediction_matches_dict(
        response[0], TEST_POLYGON_PREDICTIONS[0]
    )


def test_keypoints_pred_upload_update(model_run):
    prediction = KeypointsPrediction.from_json(TEST_KEYPOINTS_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_KEYPOINTS_PREDICTIONS[1])
    prediction_update_params["annotation_id"] = TEST_KEYPOINTS_PREDICTIONS[0][
        "annotation_id"
    ]
    prediction_update_params["reference_id"] = TEST_KEYPOINTS_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = KeypointsPrediction.from_json(prediction_update_params)
    response = model_run.predict(annotations=[prediction_update], update=True)

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["keypoints"]
    assert len(response) == 1
    assert_keypoints_prediction_matches_dict(
        response[0], prediction_update_params
    )


def test_keypoints_pred_upload_ignore(model_run):
    prediction = KeypointsPrediction.from_json(TEST_KEYPOINTS_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_KEYPOINTS_PREDICTIONS[1])
    prediction_update_params["annotation_id"] = TEST_KEYPOINTS_PREDICTIONS[0][
        "annotation_id"
    ]
    prediction_update_params["reference_id"] = TEST_KEYPOINTS_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = KeypointsPrediction.from_json(prediction_update_params)
    # Default behavior is ignore.
    response = model_run.predict(annotations=[prediction_update])

    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)["keypoints"]
    assert len(response) == 1
    assert_keypoints_prediction_matches_dict(
        response[0], TEST_KEYPOINTS_PREDICTIONS[0]
    )


def test_category_pred_upload_update(model_run):
    prediction = CategoryPrediction.from_json(TEST_CATEGORY_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_CATEGORY_PREDICTIONS[1])
    prediction_update_params["reference_id"] = TEST_CATEGORY_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = CategoryPrediction.from_json(prediction_update_params)
    response = model_run.predict(annotations=[prediction_update], update=True)

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["category"]
    assert len(response) == 1
    assert_category_prediction_matches_dict(
        response[0], prediction_update_params
    )


def test_category_pred_upload_ignore(model_run):
    prediction = CategoryPrediction.from_json(TEST_CATEGORY_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_CATEGORY_PREDICTIONS[1])
    prediction_update_params["reference_id"] = TEST_CATEGORY_PREDICTIONS[0][
        "reference_id"
    ]

    prediction_update = CategoryPrediction.from_json(prediction_update_params)
    # Default behavior is ignore.
    response = model_run.predict(annotations=[prediction_update])

    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)["category"]
    assert len(response) == 1
    assert_category_prediction_matches_dict(
        response[0], TEST_CATEGORY_PREDICTIONS[0]
    )


def test_default_category_pred_upload_update(model_run):
    prediction = CategoryPrediction.from_json(
        TEST_DEFAULT_CATEGORY_PREDICTIONS[0]
    )
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_DEFAULT_CATEGORY_PREDICTIONS[1])
    prediction_update_params[
        "reference_id"
    ] = TEST_DEFAULT_CATEGORY_PREDICTIONS[0]["reference_id"]

    prediction_update = CategoryPrediction.from_json(prediction_update_params)
    response = model_run.predict(annotations=[prediction_update], update=True)

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["category"]
    assert len(response) == 1
    assert_category_prediction_matches_dict(
        response[0], prediction_update_params
    )


def test_default_category_pred_upload_ignore(model_run):
    prediction = CategoryPrediction.from_json(
        TEST_DEFAULT_CATEGORY_PREDICTIONS[0]
    )
    response = model_run.predict(annotations=[prediction])

    assert response["predictions_processed"] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_DEFAULT_CATEGORY_PREDICTIONS[1])
    prediction_update_params[
        "reference_id"
    ] = TEST_DEFAULT_CATEGORY_PREDICTIONS[0]["reference_id"]

    prediction_update = CategoryPrediction.from_json(prediction_update_params)
    # Default behavior is ignore.
    response = model_run.predict(annotations=[prediction_update])

    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)["category"]
    assert len(response) == 1
    assert_category_prediction_matches_dict(
        response[0], TEST_DEFAULT_CATEGORY_PREDICTIONS[0]
    )


def test_mixed_pred_upload(model_run: ModelRun):
    prediction_semseg = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    prediction_line = LinePrediction.from_json(TEST_LINE_PREDICTIONS[0])
    prediction_polygon = PolygonPrediction.from_json(
        TEST_POLYGON_PREDICTIONS[0]
    )
    prediction_keypoints = KeypointsPrediction.from_json(
        TEST_KEYPOINTS_PREDICTIONS[0]
    )
    prediction_category = CategoryPrediction.from_json(
        TEST_CATEGORY_PREDICTIONS[0]
    )
    prediction_bbox = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = model_run.predict(
        annotations=[
            prediction_semseg,
            prediction_line,
            prediction_polygon,
            prediction_keypoints,
            prediction_category,
            prediction_bbox,
        ]
    )

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 6
    assert response["predictions_ignored"] == 0

    all_predictions = model_run.ungrouped_export()
    assert_box_prediction_matches_dict(
        all_predictions["box"][0], TEST_BOX_PREDICTIONS[0]
    )
    assert_line_prediction_matches_dict(
        all_predictions["line"][0], TEST_LINE_PREDICTIONS[0]
    )
    assert_polygon_prediction_matches_dict(
        all_predictions["polygon"][0], TEST_POLYGON_PREDICTIONS[0]
    )
    assert_keypoints_prediction_matches_dict(
        all_predictions["keypoints"][0], TEST_KEYPOINTS_PREDICTIONS[0]
    )
    assert_segmentation_annotation_matches_dict(
        all_predictions["segmentation"][0], TEST_SEGMENTATION_PREDICTIONS[0]
    )
    assert_category_prediction_matches_dict(
        all_predictions["category"][0], TEST_CATEGORY_PREDICTIONS[0]
    )


@pytest.mark.integration
def test_mixed_pred_upload_async(model_run: ModelRun):
    prediction_semseg = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    prediction_polygon = PolygonPrediction.from_json(
        TEST_POLYGON_PREDICTIONS[0]
    )
    prediction_category = CategoryPrediction.from_json(
        TEST_CATEGORY_PREDICTIONS[0]
    )
    prediction_bbox = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    job: AsyncJob = model_run.predict(
        annotations=[
            prediction_semseg,
            prediction_polygon,
            prediction_category,
            prediction_bbox,
        ],
        asynchronous=True,
    )
    job.sleep_until_complete()

    status = job.status()
    assert status["job_id"] == job.job_id
    assert status["status"] == "Completed"
    assert status["job_progress"] == "1.00"


@pytest.mark.integration
def test_mixed_pred_upload_async_with_error(model_run: ModelRun):
    prediction_semseg = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    prediction_polygon = PolygonPrediction.from_json(
        TEST_POLYGON_PREDICTIONS[0]
    )
    prediction_category = CategoryPrediction.from_json(
        TEST_CATEGORY_PREDICTIONS[0]
    )
    prediction_bbox = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    prediction_bbox.reference_id = "fake_garbage"

    job: AsyncJob = model_run.predict(
        annotations=[
            prediction_semseg,
            prediction_polygon,
            prediction_category,
            prediction_bbox,
        ],
        asynchronous=True,
    )
    job.sleep_until_complete()

    status = job.status()
    assert status["job_id"] == job.job_id
    assert status["status"] == "Completed"
    assert status["job_progress"] == "1.00"

    assert prediction_bbox.reference_id in str(job.errors())


@pytest.mark.integration
def test_default_category_pred_upload_async(model_run: ModelRun):
    prediction_default_category = CategoryPrediction.from_json(
        TEST_DEFAULT_CATEGORY_PREDICTIONS[0]
    )
    job: AsyncJob = model_run.predict(
        annotations=[
            prediction_default_category,
        ],
        asynchronous=True,
    )
    job.sleep_until_complete()

    status = job.status()
    assert status["job_id"] == job.job_id
    assert status["status"] == "Completed"
    assert status["job_progress"] == "1.00"


@pytest.mark.integration
def test_non_existent_taxonomy_category_pred_upload_async(model_run: ModelRun):
    prediction = CategoryPrediction.from_json(
        TEST_NONEXISTENT_TAXONOMY_CATEGORY_PREDICTION[0]
    )
    error_msg = f'Input validation failed: Taxonomy {TEST_NONEXISTENT_TAXONOMY_CATEGORY_PREDICTION[0]["taxonomy_name"]} does not exist in dataset {model_run.dataset_id}, or label {prediction.label} does not exist in the taxonomy {TEST_NONEXISTENT_TAXONOMY_CATEGORY_PREDICTION[0]["taxonomy_name"]}.'
    try:
        job: AsyncJob = model_run.predict(
            annotations=[
                prediction,
            ],
            asynchronous=True,
        )
        job.sleep_until_complete()

    except JobError:
        assert error_msg in job.errors()[-1]

    status = job.status()
    assert status["job_id"] == job.job_id
    assert status["status"] == "Errored"
    assert status["job_progress"] == "0.00"


@pytest.mark.integration
def test_box_pred_upload_embedding_async(CLIENT, model_run):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS_EMBEDDINGS[0])
    job = model_run.predict(annotations=[prediction], asynchronous=True)

    status = job.status()
    assert status["job_id"] == job.job_id
    assert status["status"] == "Running"


def test_prediction_duplicate_ids_fail(dataset, model, model_run):
    box_pred = BoxPrediction(**TEST_BOX_PREDICTIONS_EMBEDDINGS[0])
    predictions = [box_pred, box_pred]

    with pytest.raises(DuplicateIDError):
        dataset.upload_predictions(model, predictions=predictions)

    with pytest.raises(DuplicateIDError):
        model_run.predict(annotations=predictions)
