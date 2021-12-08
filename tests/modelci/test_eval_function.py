from nucleus.modelci import EvaluationCriterion


def test_list_eval_functions(CLIENT):
    eval_functions = CLIENT.modelci.eval_functions
    criteria = eval_functions.bbox_iou() > 0.5
    assert isinstance(criteria, EvaluationCriterion)
