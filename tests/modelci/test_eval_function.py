from nucleus.modelci import EvaluationCriteria


def test_list_eval_functions(CLIENT):
    eval_functions = CLIENT.modelci.eval_functions
    criteria = eval_functions.iou() > 0.5
    assert isinstance(criteria, EvaluationCriteria)
