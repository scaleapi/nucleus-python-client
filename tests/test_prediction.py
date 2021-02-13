import pytest

from helpers import (
    TEST_DATASET_NAME,
    TEST_MODEL_NAME,
    TEST_MODEL_REFERENCE,
    TEST_IMG_URLS,
    TEST_BOX_PREDICTIONS,
    TEST_POLYGON_PREDICTIONS,
    reference_id_from_url,
    assert_box_prediction_matches_dict,
    assert_polygon_prediction_matches_dict,
)

from nucleus import BoxPrediction, PolygonPrediction, DatasetItem
from nucleus.constants import ERROR_PAYLOAD

@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(DatasetItem(
            image_location=url,
            reference_id=reference_id_from_url(url),
        ))

    response = ds.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()

    model = CLIENT.add_model(
        name=TEST_MODEL_NAME,
        reference_id=TEST_MODEL_NAME
    )


    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {}


def test_box_pred_upload(dataset):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = dataset.annotate(predictions=[prediction])

    assert response['dataset_id'] == dataset.id
    assert response['predictions_processed'] == 1

    response = dataset.refloc(prediction.reference_id)['predictions']
    assert len(response) == 1
    response_prediction = response[0]
    assert_box_prediction_matches_dict(response_prediction, TEST_BOX_PREDICTIONS[0])


def test_polygon_pred_upload(dataset):
    prediction = PolygonPrediction(**TEST_POLYGON_PREDICTIONS[0])
    response = dataset.annotate(predictions=[prediction])

    assert response['dataset_id'] == dataset.id
    assert response['predictions_processed'] == 1

    response = dataset.refloc(prediction.reference_id)['predictions']
    assert len(response) == 1
    response_prediction = response[0]
    print(response_prediction)
    assert_polygon_prediction_matches_dict(response_prediction, TEST_POLYGON_PREDICTIONS[0])


def test_box_pred_upload_update(dataset):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = dataset.annotate(predictions=[prediction])

    assert response['predictions_processed'] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_BOX_PREDICTIONS[1])
    prediction_update_params['annotation_id'] = TEST_BOX_PREDICTIONS[0]['annotation_id']
    prediction_update_params['reference_id'] = TEST_BOX_PREDICTIONS[0]['reference_id']

    prediction_update = BoxPrediction(**prediction_update_params)
    response = dataset.annotate(predictions=[prediction_update], update=True)

    assert response['predictions_processed'] == 1

    response = dataset.refloc(prediction.reference_id)['predictions']
    assert len(response) == 1
    response_prediction = response[0]
    assert_box_prediction_matches_dict(response_prediction, prediction_update_params)


def test_box_pred_upload_ignore(dataset):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = dataset.annotate(predictions=[prediction])

    assert response['predictions_processed'] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_BOX_PREDICTIONS[1])
    prediction_update_params['annotation_id'] = TEST_BOX_PREDICTIONS[0]['annotation_id']
    prediction_update_params['reference_id'] = TEST_BOX_PREDICTIONS[0]['reference_id']
    prediction_update = BoxPrediction(**prediction_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(predictions=[prediction_update])

    assert response['predictions_processed'] == 1

    response = dataset.refloc(prediction.reference_id)['predictions']
    assert len(response) == 1
    response_prediction = response[0]
    assert_box_prediction_matches_dict(response_prediction, TEST_BOX_PREDICTIONS[0])


def test_polygon_pred_upload_update(dataset):
    prediction = PolygonPrediction(**TEST_POLYGON_PREDICTIONS[0])
    response = dataset.annotate(predictions=[prediction])

    assert response['predictions_processed'] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_POLYGON_PREDICTIONS[1])
    prediction_update_params['annotation_id'] = TEST_POLYGON_PREDICTIONS[0]['annotation_id']
    prediction_update_params['reference_id'] = TEST_POLYGON_PREDICTIONS[0]['reference_id']

    prediction_update = PolygonPrediction(**prediction_update_params)
    response = dataset.annotate(predictions=[prediction_update], update=True)

    assert response['predictions_processed'] == 1

    response = dataset.refloc(prediction.reference_id)['predictions']
    assert len(response) == 1
    response_prediction = response[0]
    assert_polygon_prediction_matches_dict(response_prediction, prediction_update_params)


def test_polygon_pred_upload_ignore(dataset):
    prediction = PolygonPrediction(**TEST_POLYGON_PREDICTIONS[0])
    response = dataset.annotate(predictions=[prediction])

    assert response['predictions_processed'] == 1

    # Copy so we don't modify the original.
    prediction_update_params = dict(TEST_POLYGON_PREDICTIONS[1])
    prediction_update_params['annotation_id'] = TEST_POLYGON_PREDICTIONS[0]['annotation_id']
    prediction_update_params['reference_id'] = TEST_POLYGON_PREDICTIONS[0]['reference_id']

    prediction_update = PolygonPrediction(**prediction_update_params)
    # Default behavior is ignore.
    response = dataset.annotate(predictions=[prediction_update])

    assert response['predictions_processed'] == 1

    response = dataset.refloc(prediction.reference_id)['predictions']
    assert len(response) == 1
    response_prediction = response[0]
    assert_polygon_prediction_matches_dict(response_prediction, TEST_POLYGON_PREDICTIONS[0])