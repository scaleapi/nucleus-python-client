import time

import pytest

from nucleus.annotation import BoxAnnotation
from nucleus.validate import CreateScenarioTestError
from nucleus.validate.constants import EntityLevel
from nucleus.validate.data_transfer_objects.scenario_test_evaluations import (
    EvaluationResult,
)
from nucleus.validate.scenario_test import ScenarioTest
from tests.helpers import (
    EVAL_FUNCTION_COMPARISON,
    EVAL_FUNCTION_NAME,
    TEST_MODEL_NAME,
    TEST_SCENE_BOX_ANNS_WITH_TRACK,
    TEST_SCENE_BOX_PREDS_WITH_TRACK,
    TEST_TRACK_REFERENCE_ID,
    get_uuid,
)


def test_scenario_test_metric_creation(CLIENT, annotations, scenario_test):
    # create some dataset_items for the scenario test to reference
    iou = CLIENT.validate.eval_functions.bbox_iou
    scenario_test_metric = scenario_test.add_eval_function(iou)
    assert scenario_test_metric.scenario_test_id == scenario_test.id
    assert scenario_test_metric.eval_function_id
    assert scenario_test_metric.threshold is None
    assert (
        scenario_test_metric.threshold_comparison == EVAL_FUNCTION_COMPARISON
    )

    criteria = scenario_test.get_eval_functions()
    assert isinstance(criteria, list)
    assert scenario_test_metric in criteria


@pytest.mark.xfail(
    reason="Race-condition with other tests and __post_init__ pattern. Need to refactor."
)
def test_list_scenario_test(CLIENT, test_slice, annotations):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique

    e = CLIENT.validate.eval_functions
    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_functions=[e.bbox_iou()],
    )

    scenario_tests = CLIENT.validate.scenario_tests
    assert all(
        isinstance(scenario_test, ScenarioTest)
        for scenario_test in scenario_tests
    )
    assert scenario_test in scenario_tests

    CLIENT.validate.delete_scenario_test(scenario_test.id)


def test_scenario_test_get_dataset_items(
    CLIENT,
    test_slice,
    slice_items,
):
    # Arrange
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    expected_items_locations = [item.image_location for item in slice_items]

    # Act
    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_functions=[CLIENT.validate.eval_functions.bbox_iou()],
    )
    actual_items_locations = [
        item.image_location for item in scenario_test.get_items()
    ]

    # Assert
    assert set(actual_items_locations).issubset(expected_items_locations)

    # Clean
    CLIENT.validate.delete_scenario_test(scenario_test.id)


def test_scenario_test_get_scenes(
    CLIENT,
    test_scene_slice,
    slice_scenes,
):
    # Arrange
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    expected_scene_reference_ids = [
        scene.reference_id for scene in slice_scenes
    ]

    # Act
    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_scene_slice.id,
        evaluation_functions=[CLIENT.validate.eval_functions.bbox_iou()],
    )
    actual_scene_reference_ids = [
        scene.reference_id
        for scene in scenario_test.get_items(level=EntityLevel.SCENE)
    ]

    # Assert
    assert set(actual_scene_reference_ids).issubset(
        expected_scene_reference_ids
    )

    # Clean
    CLIENT.validate.delete_scenario_test(scenario_test.id)


def test_scenario_test_get_tracks(
    CLIENT, populated_scene_dataset, test_scene_slice, annotations
):
    # Arrange
    test_name = "scenario_test_" + get_uuid()
    expected_track_reference_ids = list(
        set(
            [
                ann["track_reference_id"]
                for ann in TEST_SCENE_BOX_ANNS_WITH_TRACK
            ]
        )
    )
    annotations = [
        BoxAnnotation.from_json(ann) for ann in TEST_SCENE_BOX_ANNS_WITH_TRACK
    ]
    model_reference = "model_" + str(time.time())
    model = CLIENT.create_model(TEST_MODEL_NAME, model_reference)
    populated_scene_dataset.upload_predictions(
        model=model,
        predictions=[
            BoxAnnotation.from_json(ann)
            for ann in TEST_SCENE_BOX_PREDS_WITH_TRACK
        ],
        update=False,
        asynchronous=False,
    )
    populated_scene_dataset.annotate(
        annotations=annotations,
        update=False,
        asynchronous=False,
    )
    try:
        CLIENT.validate.create_external_eval_function(
            name=EVAL_FUNCTION_NAME,
            level=EntityLevel.TRACK,
        )
    except Exception:  # pylint: disable=W0703
        # Ignore external eval function already created
        pass
    all_external_fns = CLIENT.validate.eval_functions.external_functions
    eval_fn_config = all_external_fns[EVAL_FUNCTION_NAME]
    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_scene_slice.id,
        evaluation_functions=[eval_fn_config],
    )
    scenario_test.upload_external_evaluation_results(
        eval_fn_config,
        [
            EvaluationResult(
                track_ref_id=TEST_TRACK_REFERENCE_ID,
                score=1,
                weight=1,
            ),
        ],
        model.id,
    )

    # Act
    actual_track_reference_ids = [
        track.reference_id
        for track in scenario_test.get_items(level=EntityLevel.TRACK)
    ]

    # Assert
    assert len(actual_track_reference_ids) == len(expected_track_reference_ids)
    assert set(actual_track_reference_ids).issubset(
        expected_track_reference_ids
    )

    # Clean
    CLIENT.validate.delete_scenario_test(scenario_test.id)
    response = CLIENT.delete_model(model.id)
    assert "msg" in response
    assert response["msg"] == "Model deletion running in an async job"


def test_no_criteria_raises_error(CLIENT, test_slice, annotations):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    with pytest.raises(CreateScenarioTestError):
        CLIENT.validate.create_scenario_test(
            name=test_name,
            slice_id=test_slice.id,
            evaluation_functions=[],
        )


def test_scenario_test_set_metric_threshold(
    CLIENT, annotations, scenario_test
):
    # create some dataset_items for the scenario test to reference
    threshold = 0.5
    scenario_test_metrics = scenario_test.get_eval_functions()
    metric = scenario_test_metrics[0]
    assert metric
    metric.set_threshold(threshold)
    assert metric.threshold == threshold


def test_scenario_test_set_model_baseline(CLIENT, annotations, scenario_test):
    # create some dataset_items for the scenario test to reference
    with pytest.raises(Exception):
        scenario_test.set_baseline_model("nonexistent_model_id")


def test_passing_eval_arguments(CLIENT, test_slice, annotations):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_functions=[
            CLIENT.validate.eval_functions.bbox_iou(iou_threshold=0.5)
        ],
    )
