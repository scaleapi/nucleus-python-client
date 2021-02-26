import pytest

from helpers import (
    TEST_DATASET_NAME,
    TEST_MODEL_NAME,
    TEST_MODEL_REFERENCE,
    TEST_MODEL_RUN,
    TEST_IMG_URLS,
    TEST_BOX_PREDICTIONS,
    TEST_POLYGON_PREDICTIONS,
    TEST_SEGMENTATION_PREDICTIONS,
    reference_id_from_url,
    assert_box_prediction_matches_dict,
    assert_polygon_prediction_matches_dict,
)

from nucleus import (
    BoxPrediction,
    PolygonPrediction,
    SegmentationPrediction,
    DatasetItem,
)
from nucleus.constants import ERROR_PAYLOAD


@pytest.fixture()
def model_run(CLIENT):
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

    model = CLIENT.add_model(
        name=TEST_MODEL_NAME, reference_id=TEST_MODEL_REFERENCE
    )

    run = model.create_run(name=TEST_MODEL_RUN, dataset=ds, predictions=[])

    yield run

    response = CLIENT.delete_dataset(ds.id)
    assert response == {}
    response = CLIENT.delete_model(model.id)
    assert response == {}


def test_box_pred_upload(model_run):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)
    assert len(response) == 1
    assert_box_prediction_matches_dict(response[0], TEST_BOX_PREDICTIONS[0])


def test_polygon_pred_upload(model_run):
    prediction = PolygonPrediction(**TEST_POLYGON_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_ignored"] == 0
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)
    assert len(response) == 1
    assert_polygon_prediction_matches_dict(
        response[0], TEST_POLYGON_PREDICTIONS[0]
    )


def test_segmentation_pred_upload(model_run):
    prediction = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0


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

    response = model_run.refloc(prediction.reference_id)
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

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)
    assert len(response) == 1
    assert_box_prediction_matches_dict(response[0], TEST_BOX_PREDICTIONS[0])


def test_polygon_pred_upload_update(model_run):
    prediction = PolygonPrediction(**TEST_POLYGON_PREDICTIONS[0])
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

    prediction_update = PolygonPrediction(**prediction_update_params)
    response = model_run.predict(annotations=[prediction_update], update=True)

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)
    assert len(response) == 1
    assert_polygon_prediction_matches_dict(
        response[0], prediction_update_params
    )


def test_polygon_pred_upload_ignore(model_run):
    prediction = PolygonPrediction(**TEST_POLYGON_PREDICTIONS[0])
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

    prediction_update = PolygonPrediction(**prediction_update_params)
    # Default behavior is ignore.
    response = model_run.predict(annotations=[prediction_update])

    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)
    assert len(response) == 1
    assert_polygon_prediction_matches_dict(
        response[0], TEST_POLYGON_PREDICTIONS[0]
    )


def test_mixed_pred_upload(model_run):
    prediction_semseg = SegmentationPrediction.from_json(
        TEST_SEGMENTATION_PREDICTIONS[0]
    )
    prediction_polygon = PolygonPrediction(**TEST_POLYGON_PREDICTIONS[0])
    prediction_bbox = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = model_run.predict(
        annotations=[prediction_semseg, prediction_polygon, prediction_bbox]
    )

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 3
    assert response["predictions_ignored"] == 0
