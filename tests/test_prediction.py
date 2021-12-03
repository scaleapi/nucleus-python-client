import os
import time

import pytest

from nucleus import (
    BoxPrediction,
    CategoryPrediction,
    DatasetItem,
    ModelRun,
    Point,
    PolygonPrediction,
    Segment,
    SegmentationPrediction,
)
from nucleus.constants import ERROR_PAYLOAD
from nucleus.job import AsyncJob

from .helpers import (
    TEST_BOX_PREDICTIONS,
    TEST_CATEGORY_PREDICTIONS,
    TEST_DATASET_NAME,
    TEST_DEFAULT_CATEGORY_PREDICTIONS,
    TEST_IMG_URLS,
    TEST_MODEL_NAME,
    TEST_MODEL_RUN,
    TEST_POLYGON_PREDICTIONS,
    TEST_SEGMENTATION_PREDICTIONS,
    assert_box_prediction_matches_dict,
    assert_category_prediction_matches_dict,
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

    model = CLIENT.add_model(
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


def test_polygon_pred_upload(model_run):
    prediction = PolygonPrediction.from_json(TEST_POLYGON_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_ignored"] == 0
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["polygon"]
    assert len(response) == 1
    assert_polygon_prediction_matches_dict(
        response[0], TEST_POLYGON_PREDICTIONS[0]
    )


def test_category_pred_upload(model_run):
    prediction = CategoryPrediction.from_json(TEST_CATEGORY_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_ignored"] == 0
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
    assert response["predictions_ignored"] == 0
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["category"]
    assert len(response) == 1
    assert_category_prediction_matches_dict(
        response[0], TEST_DEFAULT_CATEGORY_PREDICTIONS[0]
    )


def test_segmentation_pred_upload(model_run):
    prediction = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["segmentation"]
    assert isinstance(response[0], SegmentationPrediction)

    assert_segmentation_annotation_matches_dict(
        response[0], TEST_SEGMENTATION_PREDICTIONS[0]
    )


def test_segmentation_pred_upload_ignore(model_run):
    prediction = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    response1 = model_run.predict(annotations=[prediction])

    assert response1["predictions_processed"] == 1

    # Upload Duplicate annotation
    response = model_run.predict(annotations=[prediction])
    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1


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
    prediction_polygon = PolygonPrediction.from_json(
        TEST_POLYGON_PREDICTIONS[0]
    )
    prediction_category = CategoryPrediction.from_json(
        TEST_CATEGORY_PREDICTIONS[0]
    )
    prediction_bbox = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = model_run.predict(
        annotations=[
            prediction_semseg,
            prediction_polygon,
            prediction_category,
            prediction_bbox,
        ]
    )

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 4
    assert response["predictions_ignored"] == 0

    all_predictions = model_run.ungrouped_export()
    assert_box_prediction_matches_dict(
        all_predictions["box"][0], TEST_BOX_PREDICTIONS[0]
    )
    assert_polygon_prediction_matches_dict(
        all_predictions["polygon"][0], TEST_POLYGON_PREDICTIONS[0]
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

    assert job.status() == {
        "job_id": job.job_id,
        "status": "Completed",
        "message": {
            "prediction_upload": {
                "epoch": 1,
                "total": 3,
                "errored": 0,
                "ignored": 0,
                "datasetId": model_run.dataset_id,
                "processed": 3,
            },
            "segmentation_upload": {
                "ignored": 0,
                "n_errors": 0,
                "processed": 1,
            },
        },
        "job_progress": "1.00",
        "completed_steps": 4,
        "total_steps": 4,
    }


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

    assert job.status() == {
        "job_id": job.job_id,
        "status": "Completed",
        "message": {
            "prediction_upload": {
                "epoch": 1,
                "total": 3,
                "errored": 1,
                "ignored": 0,
                "datasetId": model_run.dataset_id,
                "processed": 2,
            },
            "segmentation_upload": {
                "ignored": 0,
                "n_errors": 0,
                "processed": 1,
            },
        },
        "job_progress": "1.00",
        "completed_steps": 4,
        "total_steps": 4,
    }

    assert "Item with id fake_garbage doesn" in str(job.errors())


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

    assert job.status() == {
        "job_id": job.job_id,
        "status": "Completed",
        "message": {
            "prediction_upload": {
                "epoch": 1,
                "total": 1,
                "errored": 0,
                "ignored": 0,
                "datasetId": model_run.dataset_id,
                "processed": 1,
            },
        },
        "job_progress": "1.00",
        "completed_steps": 1,
        "total_steps": 1,
    }
