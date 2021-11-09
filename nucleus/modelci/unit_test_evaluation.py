from enum import Enum
from dataclasses import dataclass
from typing import List
from typing_extensions import TypedDict

from nucleus import NucleusClient
from nucleus.constants import MODEL_ID_KEY

from .constants import (
    UNIT_TEST_EVAL_STATUS_KEY,
    UNIT_TEST_ID_KEY,
    EVAL_FUNCTION_ID_KEY,
    UNIT_TEST_PASS_KEY,
    UNIT_TEST_RESULT_KEY,
)


@dataclass
class UnitTestEvaluationStatus(str, Enum):
    PENDING = ("pending",)
    STARTED = ("started",)
    COMPLETED = ("completed",)
    ERRORED = ("errored",)


class UnitTestEvaluationInfo(TypedDict):
    unit_test_id: str
    eval_function_id: str
    model_id: str
    status: UnitTestEvaluationStatus
    result: float
    passed: bool


class UnitTestItemEvaluationInfo(TypedDict):
    evaluation_id: str
    evaluation_id: str
    unit_test_id: str
    eval_function_id: str
    dataset_item_id: str
    result: float
    passed: bool


class UnitTestEvaluation:
    """
    Model CI Unit Test Evaluation.
    """

    def __init__(self, evaluation_id: str, client: NucleusClient):
        self.id = evaluation_id
        self._client = client
        info = self._client.get_unit_test_eval_info(self.id)
        self.unit_test_id = info[UNIT_TEST_ID_KEY]
        self.eval_function_id = info[EVAL_FUNCTION_ID_KEY]
        self.model_id = info[MODEL_ID_KEY]
        self.status = info[UNIT_TEST_EVAL_STATUS_KEY]
        self.result = info[UNIT_TEST_RESULT_KEY]
        self.passed = info[UNIT_TEST_PASS_KEY]

    def get_item_evaluations(self) -> List[UnitTestItemEvaluationInfo]:
        """
        Get item evaluation results for this unit test evaluation.
        :return: List[UnitTestItemEvaluationInfo]
        """
        return self._client.get_unit_test_item_eval_results(
            self.id, self.unit_test_id
        )
