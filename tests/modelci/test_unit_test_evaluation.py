import pytest

from nucleus import BoxAnnotation, BoxPrediction
from nucleus.job import AsyncJob
from nucleus.modelci.unit_test_evaluation import (
    UnitTestEvaluation,
    UnitTestItemEvaluation,
)
from tests.helpers import (
    EVAL_FUNCTION_THRESHOLD,
    TEST_BOX_ANNOTATIONS,
    TEST_BOX_PREDICTIONS,
)


@pytest.mark.integration
def test_unit_test_evaluation(CLIENT, dataset, model, unit_test):
    annotation = BoxAnnotation(**TEST_BOX_ANNOTATIONS[0])
    dataset.annotate(annotations=[annotation])
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[0])
    dataset.upload_predictions(model, [prediction])

    iou = CLIENT.modelci.eval_functions.iou
    unit_test.add_criteria(iou() > EVAL_FUNCTION_THRESHOLD)

    job: AsyncJob = CLIENT.modelci.evaluate_model_on_unit_tests(
        model.id, [unit_test.name]
    )
    job.sleep_until_complete()

    evaluations = unit_test.get_eval_history()
    assert isinstance(evaluations, list)
    assert len(evaluations) == 1
    assert isinstance(evaluations[0], UnitTestEvaluation)
    assert evaluations[0].unit_test_id == unit_test.id
    assert evaluations[0].model_id == model.id

    item_evaluations = evaluations[0].item_evals
    assert isinstance(item_evaluations, list)
    assert len(item_evaluations) == 1  # 1 item in slice
    assert isinstance(item_evaluations[0], UnitTestItemEvaluation)
    assert all(
        eval.evaluation_id == evaluations[0].id for eval in item_evaluations
    )
    assert all(eval.unit_test_id == unit_test.id for eval in item_evaluations)
