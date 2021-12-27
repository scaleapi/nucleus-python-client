import pytest

from nucleus.job import AsyncJob
from nucleus.modelci.unit_test_evaluation import (
    UnitTestEvaluation,
    UnitTestItemEvaluation,
)
from tests.helpers import EVAL_FUNCTION_THRESHOLD, get_uuid
from tests.modelci.helpers import create_predictions


@pytest.mark.skip("Assigned slice has no 2D annotations")
@pytest.mark.integration
def test_unit_test_evaluation(
    CLIENT, modelci_dataset, model, unit_test, annotations, predictions
):
    iou = CLIENT.modelci.eval_functions.bbox_iou
    # NOTE: Another criterion is defined in the unit_test fixture
    unit_test.add_criterion(iou() > EVAL_FUNCTION_THRESHOLD)

    job: AsyncJob = CLIENT.modelci.evaluate_model_on_unit_tests(
        model.id, [unit_test.name]
    )
    job.sleep_until_complete()

    criteria = unit_test.get_criteria()
    evaluations = unit_test.get_eval_history()
    assert isinstance(evaluations, list)
    assert len(evaluations) == len(criteria)
    assert all(
        isinstance(evaluation, UnitTestEvaluation)
        for evaluation in evaluations
    )
    assert all(
        evaluation.unit_test_id == unit_test.id for evaluation in evaluations
    )
    assert all(evaluation.model_id == model.id for evaluation in evaluations)

    unit_test_slice = CLIENT.get_slice(unit_test.slice_id)
    item_evaluations = evaluations[0].item_evals
    assert isinstance(item_evaluations, list)
    assert len(item_evaluations) == len(
        unit_test_slice.items_and_annotations()
    )
    assert isinstance(item_evaluations[0], UnitTestItemEvaluation)
    assert all(
        eval.evaluation_id == evaluations[0].id for eval in item_evaluations
    )
    assert all(eval.unit_test_id == unit_test.id for eval in item_evaluations)


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Missing predictions is currently treated as failure in evaluation."
)
@pytest.mark.skip
def test_unit_test_evaluation_no_prediction_for_last_item(
    # TODO(gunnar): Remove this slow integration tests after this is confirmed and tested on the evaluation side.
    #  there's no reason doing unit testing for evaluation here.
    CLIENT,
    modelci_dataset,
    unit_test,
    annotations,
):
    uuid = get_uuid()
    model = CLIENT.create_model(
        f"[Model CI Test] {uuid}", reference_id=f"model_ci_{uuid}"
    )
    create_predictions(modelci_dataset, model, annotations[:-1])
    iou = CLIENT.modelci.eval_functions.bbox_iou
    # NOTE: Another criterion is defined in the unit_test fixture
    unit_test.add_criterion(iou() > EVAL_FUNCTION_THRESHOLD)

    job: AsyncJob = CLIENT.modelci.evaluate_model_on_unit_tests(
        model.id, [unit_test.name]
    )
    job.sleep_until_complete()

    criteria = unit_test.get_criteria()
    evaluations = unit_test.get_eval_history()
    assert isinstance(evaluations, list)
    assert len(evaluations) == len(criteria)
    assert all(
        isinstance(evaluation, UnitTestEvaluation)
        for evaluation in evaluations
    )
    assert all(
        evaluation.unit_test_id == unit_test.id for evaluation in evaluations
    )
    assert all(evaluation.model_id == model.id for evaluation in evaluations)

    unit_test_slice = CLIENT.get_slice(unit_test.slice_id)
    item_evaluations = evaluations[0].item_evals
    assert isinstance(item_evaluations, list)
    assert len(item_evaluations) == len(
        unit_test_slice.items_and_annotations()
    )
    assert isinstance(item_evaluations[0], UnitTestItemEvaluation)
    assert all(
        eval.evaluation_id == evaluations[0].id for eval in item_evaluations
    )
    assert all(eval.unit_test_id == unit_test.id for eval in item_evaluations)
