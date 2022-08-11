"""Data types for Scenario Test Evaluation results."""
from dataclasses import dataclass, field
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
    connection: Connection = field(init=False, repr=False)

    @classmethod
    def from_request(cls, response, connection):
        instance = cls(response["id"])
        instance.connection = connection

        instance.scenario_test_id = response[SCENARIO_TEST_ID_KEY]
        instance.eval_function_id = response[EVAL_FUNCTION_ID_KEY]
        instance.model_id = response[MODEL_ID_KEY]
        instance.status = ScenarioTestEvaluationStatus(response[STATUS_KEY])
        instance.result = try_convert_float(response[RESULT_KEY])
        instance.passed = bool(response[PASS_KEY])
        return instance

    @property
    def item_evals(self) -> List[ScenarioTestItemEvaluation]:
        response = self.connection.make_request(
            {},
            f"validate/eval/{self.id}/info",
            requests_command=requests.get,
        )
        items_response = response[ITEM_EVAL_KEY]
        items = [
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
        return items
