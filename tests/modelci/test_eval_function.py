from nucleus.modelci.eval_function import EvalFunctionDefinition


def test_list_eval_functions(CLIENT):
    eval_functions = CLIENT.modelci.list_eval_functions()
    assert all(
        isinstance(eval_fn, EvalFunctionDefinition) for eval_fn in eval_functions
    )
