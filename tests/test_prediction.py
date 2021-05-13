import pytest
import time
from .helpers import (
    TEST_DATASET_NAME,
    TEST_MODEL_NAME,
    TEST_MODEL_RUN,
    TEST_IMG_URLS,
    TEST_BOX_ANNOTATIONS,
    TEST_BOX_PREDICTIONS,
    TEST_POLYGON_PREDICTIONS,
    TEST_SEGMENTATION_PREDICTIONS,
    reference_id_from_url,
    assert_box_prediction_matches_dict,
    assert_polygon_prediction_matches_dict,
    assert_segmentation_annotation_matches_dict,
)

from nucleus import (
    BoxAnnotation,
    BoxPrediction,
    PolygonPrediction,
    SegmentationPrediction,
    DatasetItem,
    Segment,
)
from nucleus.constants import ERROR_PAYLOAD


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


@pytest.fixture()
def model_run(CLIENT, dataset):
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
            )
        )

    response = dataset.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()

    model = CLIENT.add_model(
        name=TEST_MODEL_NAME, reference_id="model_" + str(time.time())
    )

    run = model.create_run(name=TEST_MODEL_RUN,
                           dataset=dataset, predictions=[])

    yield run
    response = CLIENT.delete_model(model.id)
    assert response == {}


def test_schema_validation(model_run, dataset):
    annotations = []
    for annotation in TEST_BOX_ANNOTATIONS:
        annotations.append(BoxAnnotation(**annotation))
    response = dataset.annotate(annotations=annotations)
    assert response["annotations_processed"] == len(annotations)

    predictions = []
    for prediction in TEST_BOX_PREDICTIONS:
        predictions.append(BoxPrediction(**prediction))
    response = model_run.predict(annotations=predictions)
    assert response["predictions_processed"] == len(TEST_BOX_PREDICTIONS)

    annotation_labels = [annotation.label for annotation in annotations]
    prediction_labels = [prediction.label for prediction in predictions]

    allowed_label_matches = [{'ground_truth_label': annotation_label, 'model_prediction_label': prediction_label}
                             for annotation_label, prediction_label in zip(annotation_labels, prediction_labels)]
    allowed_label_matches_fudged = [{'ground_truth_label': f'{annotation_label}t', 'model_prediction_label': prediction_label}
                                    for annotation_label, prediction_label in zip(annotation_labels, prediction_labels)]
    error = None
    try:
        model_run.commit(
            payload={'allowed_label_matches': allowed_label_matches_fudged})
    except Exception as e:
        error = e
    assert(error is not None)

    # If this fails, we would raise an error.
    response = model_run.commit(
        payload={'allowed_label_matches': allowed_label_matches})


def test_box_pred_upload(model_run):
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_processed"] == 1
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["box"]
    assert len(response) == 1
    assert_box_prediction_matches_dict(response[0], TEST_BOX_PREDICTIONS[0])


def test_polygon_pred_upload(model_run):
    prediction = PolygonPrediction(**TEST_POLYGON_PREDICTIONS[0])
    response = model_run.predict(annotations=[prediction])

    assert response["model_run_id"] == model_run.model_run_id
    assert response["predictions_ignored"] == 0
    assert response["predictions_ignored"] == 0

    response = model_run.refloc(prediction.reference_id)["polygon"]
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

    response = model_run.refloc(prediction.reference_id)["polygon"]
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

    assert response["predictions_processed"] == 0
    assert response["predictions_ignored"] == 1

    response = model_run.refloc(prediction.reference_id)["polygon"]
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

    response_refloc = model_run.refloc(prediction_polygon.reference_id)
    assert_box_prediction_matches_dict(
        response_refloc["box"][0], TEST_BOX_PREDICTIONS[0]
    )
    assert_polygon_prediction_matches_dict(
        response_refloc["polygon"][0], TEST_POLYGON_PREDICTIONS[0]
    )
    assert_segmentation_annotation_matches_dict(
        response_refloc["segmentation"][0], TEST_SEGMENTATION_PREDICTIONS[0]
    )
