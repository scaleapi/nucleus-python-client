import pytest

from nucleus.metrics import PolygonIOU
from nucleus.modelci import EvaluationCriterion

TEST_EVAL_FUNC_NAME = "test_eval_func"


def test_list_eval_functions(CLIENT):
    eval_functions = CLIENT.modelci.eval_functions
    criteria = eval_functions.bbox_iou() > 0.5
    assert isinstance(criteria, EvaluationCriterion)


@pytest.mark.integration
def test_eval_function_upload(CLIENT):
    metric = PolygonIOU()
    job = CLIENT.modelci.upload_eval_function(metric, TEST_EVAL_FUNC_NAME)
    job.sleep_until_complete()
    eval_functions = CLIENT.modelci.eval_functions
    assert TEST_EVAL_FUNC_NAME in eval_functions.private_functions
    assert (
        type(eval_functions.private_functions[TEST_EVAL_FUNC_NAME])
        == PolygonIOU
    )
    assert dir(eval_functions.private_functions[TEST_EVAL_FUNC_NAME]) == dir(
        metric
    )
