import pytest

from nucleus.job import AsyncJob
from nucleus.modelci.scenario_test_evaluation import (
    ScenarioTestEvaluation,
    ScenarioTestItemEvaluation,
)
from tests.helpers import EVAL_FUNCTION_THRESHOLD, get_uuid
from tests.modelci.helpers import create_predictions


@pytest.mark.skip("Assigned slice has no 2D annotations")
@pytest.mark.integration
def test_scenario_test_evaluation(
    CLIENT, modelci_dataset, model, scenario_test, annotations, predictions
):
    iou = CLIENT.modelci.eval_functions.bbox_iou
    # NOTE: Another criterion is defined in the scenario_test fixture
    scenario_test.add_criterion(iou() > EVAL_FUNCTION_THRESHOLD)

    job: AsyncJob = CLIENT.modelci.evaluate_model_on_scenario_tests(
        model.id, [scenario_test.name]
    )
    job.sleep_until_complete()

    criteria = scenario_test.get_criteria()
    evaluations = scenario_test.get_eval_history()
    assert isinstance(evaluations, list)
    assert len(evaluations) == len(criteria)
    assert all(
        isinstance(evaluation, ScenarioTestEvaluation)
        for evaluation in evaluations
    )
    assert all(
        evaluation.scenario_test_id == scenario_test.id
        for evaluation in evaluations
    )
    assert all(evaluation.model_id == model.id for evaluation in evaluations)

    scenario_test_slice = CLIENT.get_slice(scenario_test.slice_id)
    item_evaluations = evaluations[0].item_evals
    assert isinstance(item_evaluations, list)
    assert len(item_evaluations) == len(
        scenario_test_slice.items_and_annotations()
    )
    assert isinstance(item_evaluations[0], ScenarioTestItemEvaluation)
    assert all(
        eval.evaluation_id == evaluations[0].id for eval in item_evaluations
    )
    assert all(
        eval.scenario_test_id == scenario_test.id for eval in item_evaluations
    )


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Missing predictions is currently treated as failure in evaluation."
)
@pytest.mark.skip
def test_scenario_test_evaluation_no_prediction_for_last_item(
    # TODO(gunnar): Remove this slow integration tests after this is confirmed and tested on the evaluation side.
    #  there's no reason doing scenario testing for evaluation here.
    CLIENT,
    modelci_dataset,
    scenario_test,
    annotations,
):
    uuid = get_uuid()
    model = CLIENT.create_model(
        f"[Model CI Test] {uuid}", reference_id=f"model_ci_{uuid}"
    )
    create_predictions(modelci_dataset, model, annotations[:-1])
    iou = CLIENT.modelci.eval_functions.bbox_iou
    # NOTE: Another criterion is defined in the scenario_test fixture
    scenario_test.add_criterion(iou() > EVAL_FUNCTION_THRESHOLD)

    job: AsyncJob = CLIENT.modelci.evaluate_model_on_scenario_tests(
        model.id, [scenario_test.name]
    )
    job.sleep_until_complete()

    criteria = scenario_test.get_criteria()
    evaluations = scenario_test.get_eval_history()
    assert isinstance(evaluations, list)
    assert len(evaluations) == len(criteria)
    assert all(
        isinstance(evaluation, ScenarioTestEvaluation)
        for evaluation in evaluations
    )
    assert all(
        evaluation.scenario_test_id == scenario_test.id
        for evaluation in evaluations
    )
    assert all(evaluation.model_id == model.id for evaluation in evaluations)

    scenario_test_slice = CLIENT.get_slice(scenario_test.slice_id)
    item_evaluations = evaluations[0].item_evals
    assert isinstance(item_evaluations, list)
    assert len(item_evaluations) == len(
        scenario_test_slice.items_and_annotations()
    )
    assert isinstance(item_evaluations[0], ScenarioTestItemEvaluation)
    assert all(
        eval.evaluation_id == evaluations[0].id for eval in item_evaluations
    )
    assert all(
        eval.scenario_test_id == scenario_test.id for eval in item_evaluations
    )
