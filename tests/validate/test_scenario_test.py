import pytest

from nucleus.validate import CreateScenarioTestError
from nucleus.validate.scenario_test import ScenarioTest
from tests.helpers import (
    EVAL_FUNCTION_COMPARISON,
    EVAL_FUNCTION_THRESHOLD,
    get_uuid,
)


def test_scenario_test_metric_creation(CLIENT, annotations, scenario_test):
    # create some dataset_items for the scenario test to reference
    iou = CLIENT.validate.eval_functions.bbox_iou
    scenario_test_metric = scenario_test.add_criterion(
        iou() > EVAL_FUNCTION_THRESHOLD
    )
    assert scenario_test_metric.scenario_test_id == scenario_test.id
    assert scenario_test_metric.eval_function_id
    assert scenario_test_metric.threshold == EVAL_FUNCTION_THRESHOLD
    assert (
        scenario_test_metric.threshold_comparison == EVAL_FUNCTION_COMPARISON
    )

    criteria = scenario_test.get_criteria()
    assert isinstance(criteria, list)
    assert scenario_test_metric in criteria


def test_list_scenario_test(CLIENT, test_slice, annotations):
    test_name = "scenario_test_" + get_uuid()  # use uuid to make unique

    e = CLIENT.validate.eval_functions
    scenario_test = CLIENT.validate.create_scenario_test(
        name=test_name,
        slice_id=test_slice.id,
        evaluation_criteria=[e.bbox_iou() > 0.5],
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
        evaluation_criteria=[CLIENT.validate.eval_functions.bbox_iou() > 0.5],
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
            evaluation_criteria=[],
        )
