from .unit_test_evaluation import (
    UnitTestEvaluationInfo,
    UnitTestItemEvaluationInfo,
    UnitTestEvaluationStatus,
)


def format_unit_test_eval_response(response: dict) -> UnitTestEvaluationInfo:
    return {
        "unit_test_id": response["unit_test_id"],
        "eval_function_id": response["eval_function_id"],
        "model_id": response["model_id"],
        "status": UnitTestEvaluationStatus(response["status"]),
        "result": response["result"],
        "passed": response["pass"],
    }


def format_unit_test_item_eval_response(
    response: dict,
) -> UnitTestItemEvaluationInfo:
    return {
        "unit_test_id": response["unit_test_id"],
        "eval_function_id": response["eval_function_id"],
        "model_id": response["model_id"],
        "status": UnitTestEvaluationStatus(response["status"]),
        "result": response["result"],
        "passed": response["pass"],
    }
