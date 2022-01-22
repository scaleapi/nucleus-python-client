import pytest

from nucleus.metrics import PolygonIOU
from nucleus.modelci import EvaluationCriterion
from tests.helpers import get_uuid

TEST_EVAL_FUNC_NAME = "test_eval_func" + get_uuid()


def test_list_eval_functions(CLIENT):
    eval_functions = CLIENT.modelci.eval_functions
    criteria = eval_functions.bbox_iou() > 0.5
    assert isinstance(criteria, EvaluationCriterion)


@pytest.mark.integration
def test_eval_function_upload(CLIENT):
    metric = PolygonIOU()
    result = CLIENT.modelci.upload_eval_function(metric, TEST_EVAL_FUNC_NAME)
    assert TEST_EVAL_FUNC_NAME == result.name
    eval_functions = CLIENT.modelci.eval_functions
    assert TEST_EVAL_FUNC_NAME in eval_functions.private_functions
    CLIENT.modelci.delete_eval_function(result.id)
