from typing import Optional

from .unit_test_evaluation import UnitTestEvaluation, UnitTestItemEvaluation


def try_convert_float(float_str: str) -> Optional[float]:
    try:
        return float(float_str)
    except ValueError:
        return None


def format_unit_test_item_eval_response(
    response: dict,
) -> UnitTestItemEvaluation:
    return UnitTestItemEvaluation(
        evaluation_id=response["evaluation_id"],
        unit_test_id=response["unit_test_id"],
        eval_function_id=response["eval_function_id"],
        dataset_item_id=response["dataset_item_id"],
        result=try_convert_float(response["result"]),
        passed=bool(response["pass"]),
    )


def format_unit_test_eval_response(response: dict) -> UnitTestEvaluation:
    eval_response = response["unit_test_evaluation"]
    items_response = response["unit_test_item_evaluations"]
    item_evals = [
        format_unit_test_item_eval_response(res) for res in items_response
    ]
    return UnitTestEvaluation(
        id=eval_response["id"],
        unit_test_id=eval_response["unit_test_id"],
        eval_function_id=eval_response["eval_function_id"],
        model_id=eval_response["model_id"],
        status=eval_response["status"],
        result=try_convert_float(eval_response["result"]),
        passed=bool(eval_response["pass"]),
        item_evals=item_evals,
    )
