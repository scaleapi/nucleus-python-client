from enum import Enum
from dataclasses import dataclass
from typing import List


@dataclass
class UnitTestEvaluationStatus(str, Enum):
    PENDING = ("pending",)
    STARTED = ("started",)
    COMPLETED = ("completed",)
    ERRORED = ("errored",)


@dataclass
class UnitTestEvaluation:
    unit_test_id: str
    eval_function_id: str
    model_id: str
    status: UnitTestEvaluationStatus
    result: float
    passed: bool


@dataclass
class UnitTestItemEvaluation:
    evaluation_id: str
    unit_test_id: str
    eval_function_id: str
    dataset_item_id: str
    result: float
    passed: bool


@dataclass
class UnitTestEvaluation:
    id: str
    unit_test_id: str
    eval_function_id: str
    model_id: str
    status: str
    result: float
    passed: bool
    item_evals: List[UnitTestItemEvaluation]
