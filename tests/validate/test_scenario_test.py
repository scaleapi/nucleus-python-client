import pytest

from nucleus.validate import CreateScenarioTestError
from nucleus.validate.errors import InvalidEvaluationCriteria
from nucleus.validate.scenario_test import ScenarioTest
from tests.helpers import (
    EVAL_FUNCTION_COMPARISON,
    EVAL_FUNCTION_THRESHOLD,
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


def test_scenario_test_items(CLIENT, test_slice, slice_items, annotations):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique

    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_functions=[CLIENT.validate.eval_functions.bbox_iou()],
    )

    expected_items_locations = [item.image_location for item in slice_items]
    actual_items_locations = [
        item.image_location for item in scenario_test.get_items()
    ]
    assert expected_items_locations == actual_items_locations
    CLIENT.validate.delete_scenario_test(scenario_test.id)


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


def test_missing_comparison_raises_invalid_criteria(
    CLIENT, test_slice, annotations
):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    with pytest.raises(InvalidEvaluationCriteria):
        CLIENT.validate.create_scenario_test(
            name=test_name,
            slice_id=test_slice.id,
            evaluation_criteria=[
                CLIENT.validate.eval_functions.bbox_iou(iou_threshold=0.5)
            ],
        )


def test_passing_eval_arguments(CLIENT, test_slice, annotations):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique
    CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_criteria=[
            CLIENT.validate.eval_functions.bbox_iou(iou_threshold=0.5) > 0
        ],
    )
