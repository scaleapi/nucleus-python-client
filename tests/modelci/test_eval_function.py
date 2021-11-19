from nucleus.modelci.eval_function import EvalFunction


def test_list_eval_functions(CLIENT):
    eval_functions = CLIENT.modelci.list_eval_functions()
    assert all(isinstance(eval_fn, EvalFunction) for eval_fn in eval_functions)
