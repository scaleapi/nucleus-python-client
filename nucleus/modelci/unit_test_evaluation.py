from enum import Enum
from dataclasses import dataclass
from typing import List
from typing_extensions import TypedDict


@dataclass
class UnitTestEvaluationStatus(str, Enum):
    PENDING = ("pending",)
    STARTED = ("started",)
    COMPLETED = ("completed",)
    ERRORED = ("errored",)


class UnitTestEvaluation(TypedDict):
    unit_test_id: str
    eval_function_id: str
    model_id: str
    status: UnitTestEvaluationStatus
    result: float
    passed: bool


class UnitTestItemEvaluation(TypedDict):
    evaluation_id: str
    unit_test_id: str
    eval_function_id: str
    dataset_item_id: str
    result: float
    passed: bool


class UnitTestEvaluation(TypedDict):
    id: str
    unit_test_id: str
    eval_function_id: str
    model_id: str
    status: str
    result: float
    passed: bool
    item_evals: List[UnitTestItemEvaluation]
