import pytest
import uuid

from ..test_dataset import make_dataset_items
from ..helpers import (
    TEST_SLICE_NAME,
    EVAL_FUNCTION_NAME,
    EVAL_FUNCTION_THRESHOLD,
    get_uuid,
)

from nucleus.job import AsyncJob
from nucleus.modelci.unit_test_evaluation import (
    UnitTestEvaluation,
    UnitTestItemEvaluation,
)

# TODO: Move unit test to fixture once deletion is implemented
@pytest.mark.integration
def test_unit_test_evaluation(CLIENT, dataset, model):
    # create some dataset_items for the unit test to reference
    items = make_dataset_items()
    dataset.append(items)
    test_name = "unit_test_" + get_uuid()  # use uuid to make unique
    slc = dataset.create_slice(
        name=TEST_SLICE_NAME,
        reference_ids=[items[0].reference_id],
    )

    unit_test = CLIENT.modelci.create_unit_test(
        name=test_name,
        slice_id=slc.slice_id,
    )

    unit_test_metric = unit_test.add_metric(
        eval_function_name=EVAL_FUNCTION_NAME,
        threshold=EVAL_FUNCTION_THRESHOLD,
        threshold_comparison=ThresholdComparison.GREATER_THAN,
    )

    job: AsyncJob = CLIENT.modelci.evaluate_model_on_unit_tests(
        model.id, [test_name]
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
