"""Data types for Unit Test Evaluation results."""
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class UnitTestEvaluationStatus(str, Enum):
    """Job status of unit test evaluation."""

    PENDING = ("pending",)
    STARTED = ("started",)
    COMPLETED = ("completed",)
    ERRORED = ("errored",)


@dataclass
class UnitTestItemEvaluation:
    """Dataset item-level results of an evaluation of a unit test."""

    evaluation_id: str
    unit_test_id: str
    eval_function_id: str
    dataset_item_id: str
    result: Optional[float]
    passed: bool


@dataclass
class UnitTestEvaluation:
    """Results and attributes of an evaluation of a unit test."""

    # pylint: disable=too-many-instance-attributes

    id: str
    unit_test_id: str
    eval_function_id: str
    model_id: str
    status: str
    result: Optional[float]
    passed: bool
    item_evals: List[UnitTestItemEvaluation]
