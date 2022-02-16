"""Data types for Scenario Test Evaluation results."""
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import List, Optional

import requests

from nucleus.connection import Connection
from nucleus.constants import DATASET_ITEM_ID_KEY, MODEL_ID_KEY, STATUS_KEY
from nucleus.validate.constants import (
    EVAL_FUNCTION_ID_KEY,
    EVALUATION_ID_KEY,
    PASS_KEY,
    RESULT_KEY,
    SCENARIO_TEST_ID_KEY,
)

from .utils import try_convert_float

SCENARIO_TEST_EVAL_KEY = "scenario_test_evaluation"
ITEM_EVAL_KEY = "scenario_test_item_evaluations"


class ScenarioTestEvaluationStatus(Enum):
    """The Job status of scenario test evaluation."""

    PENDING = "pending"
    STARTED = "started"
    COMPLETED = "completed"
    ERRORED = "errored"


@dataclass(frozen=True)
class ScenarioTestItemEvaluation:
    """Dataset item-level results of an evaluation of a scenario test.
    Note that this class is immutable.

    Attributes:
        evaluation_id (str): The ID of the associated scenario test evaluation
        scenario_test_id (str): The ID of the associated scenario test.
        eval_function_id (str): The ID of the associated evaluation function.
        dataset_item_id (str): The ID of the dataset item of this evaluation.
        result (Optional[float]): The numerical result of the evaluation on this item.
        passed (bool): Whether the result was sufficient to pass the test for this item.
    """

    evaluation_id: str
    scenario_test_id: str
    eval_function_id: str
    dataset_item_id: str
    result: Optional[float]
    passed: bool


@dataclass
class ScenarioTestEvaluation:
    """The results and attributes of an evaluation of a scenario test.

    Attributes:
        id (str): The ID of this scenario test evaluation.
        scenario_test_id (str): The ID of the associated scenario test.
        eval_function_id (str): The ID of the associated evaluation function.
        model_id (str): The ID of the associated model.
        status (str): The status of the evaluation job.
        result (Optional[float]): The float result of the evaluation.
        passed (bool): Whether the scenario test was passed.
        item_evals (List[ScenarioTestItemEvaluation]): The individual results for each dataset item.
        connection (Connection): The connection to the Nucleus API.
    """

    # pylint: disable=too-many-instance-attributes

    id: str
    scenario_test_id: str = field(init=False)
    eval_function_id: str = field(init=False)
    model_id: str = field(init=False)
    status: ScenarioTestEvaluationStatus = field(init=False)
    result: Optional[float] = field(init=False)
    passed: bool = field(init=False)
    item_evals: List[ScenarioTestItemEvaluation] = field(init=False)
    connection: InitVar[Connection]

    def __post_init__(self, connection: Connection):
        # TODO(gunnar): Having the function call /info on every construction is too slow. The original
        #  endpoint should rather return the necessary human-readable information
        response = connection.make_request(
            {},
            f"validate/eval/{self.id}/info",
            requests_command=requests.get,
        )
        eval_response = response[SCENARIO_TEST_EVAL_KEY]
        items_response = response[ITEM_EVAL_KEY]

        self.scenario_test_id: str = eval_response[SCENARIO_TEST_ID_KEY]
        self.eval_function_id: str = eval_response[EVAL_FUNCTION_ID_KEY]
        self.model_id: str = eval_response[MODEL_ID_KEY]
        self.status: ScenarioTestEvaluationStatus = (
            ScenarioTestEvaluationStatus(eval_response[STATUS_KEY])
        )
        self.result: Optional[float] = try_convert_float(
            eval_response[RESULT_KEY]
        )
        self.passed: bool = bool(eval_response[PASS_KEY])
        self.item_evals: List[ScenarioTestItemEvaluation] = [
            ScenarioTestItemEvaluation(
                evaluation_id=res[EVALUATION_ID_KEY],
                scenario_test_id=res[SCENARIO_TEST_ID_KEY],
                eval_function_id=res[EVAL_FUNCTION_ID_KEY],
                dataset_item_id=res[DATASET_ITEM_ID_KEY],
                result=try_convert_float(res[RESULT_KEY]),
                passed=bool(res[PASS_KEY]),
            )
            for res in items_response
        ]
