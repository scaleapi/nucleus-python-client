"""Data types for Unit Test Evaluation results."""
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import List, Optional

import requests

from nucleus.connection import Connection
from nucleus.constants import DATASET_ITEM_ID_KEY, MODEL_ID_KEY, STATUS_KEY
from nucleus.modelci.constants import (
    EVAL_FUNCTION_ID_KEY,
    EVALUATION_ID_KEY,
    PASS_KEY,
    RESULT_KEY,
    UNIT_TEST_ID_KEY,
)

from .utils import try_convert_float

UNIT_TEST_EVAL_KEY = "unit_test_evaluation"
ITEM_EVAL_KEY = "unit_test_item_evaluations"


class UnitTestEvaluationStatus(Enum):
    """The Job status of unit test evaluation."""

    PENDING = "pending"
    STARTED = "started"
    COMPLETED = "completed"
    ERRORED = "errored"


@dataclass(frozen=True)
class UnitTestItemEvaluation:
    """Dataset item-level results of an evaluation of a unit test.
    Note that this class is immutable.

    Attributes:
        evaluation_id (str): The ID of the associated unit test evaluation
        unit_test_id (str): The ID of the associated unit test.
        eval_function_id (str): The ID of the associated evaluation function.
        dataset_item_id (str): The ID of the dataset item of this evaluation.
        result (Optional[float]): The numerical result of the evaluation on this item.
        passed (bool): Whether the result was sufficient to pass the test for this item.
    """

    evaluation_id: str
    unit_test_id: str
    eval_function_id: str
    dataset_item_id: str
    result: Optional[float]
    passed: bool


@dataclass
class UnitTestEvaluation:
    """The results and attributes of an evaluation of a unit test.

    Attributes:
        id (str): The ID of this unit test evaluation.
        unit_test_id (str): The ID of the associated unit test.
        eval_function_id (str): The ID of the associated evaluation function.
        model_id (str): THe ID of the associated model.
        status (str): The status of the evaluation job.
        result (Optional[float]): The float result of the evaluation.
        passed (bool): Whether the unit test was passed.
        item_evals (List[UnitTestItemEvaluation]): The individual results for each dataset item.
        connection (Connection): The connection to the Nucleus API.
    """

    # pylint: disable=too-many-instance-attributes

    id: str
    unit_test_id: str = field(init=False)
    eval_function_id: str = field(init=False)
    model_id: str = field(init=False)
    status: UnitTestEvaluationStatus = field(init=False)
    result: Optional[float] = field(init=False)
    passed: bool = field(init=False)
    item_evals: List[UnitTestItemEvaluation] = field(init=False)
    connection: InitVar[Connection]

    def __post_init__(self, connection: Connection):
        # TODO(gunnar): Having the function call /info on every construction is too slow. The original
        #  endpoint should rather return the necessary human-readable information
        response = connection.make_request(
            {},
            f"modelci/eval/{self.id}/info",
            requests_command=requests.get,
        )
        eval_response = response[UNIT_TEST_EVAL_KEY]
        items_response = response[ITEM_EVAL_KEY]

        self.unit_test_id: str = eval_response[UNIT_TEST_ID_KEY]
        self.eval_function_id: str = eval_response[EVAL_FUNCTION_ID_KEY]
        self.model_id: str = eval_response[MODEL_ID_KEY]
        self.status: UnitTestEvaluationStatus = UnitTestEvaluationStatus(
            eval_response[STATUS_KEY]
        )
        self.result: Optional[float] = try_convert_float(
            eval_response[RESULT_KEY]
        )
        self.passed: bool = bool(eval_response[PASS_KEY])
        self.item_evals: List[UnitTestItemEvaluation] = [
            UnitTestItemEvaluation(
                evaluation_id=res[EVALUATION_ID_KEY],
                unit_test_id=res[UNIT_TEST_ID_KEY],
                eval_function_id=res[EVAL_FUNCTION_ID_KEY],
                dataset_item_id=res[DATASET_ITEM_ID_KEY],
                result=try_convert_float(res[RESULT_KEY]),
                passed=bool(res[PASS_KEY]),
            )
            for res in items_response
        ]
