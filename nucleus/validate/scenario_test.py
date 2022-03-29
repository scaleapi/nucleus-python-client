"""Scenario Tests combine collections of data and evaluation metrics to accelerate model evaluation.

With Model CI Scenario Tests, an ML engineer can define a Scenario Test from critical
edge case scenarios that the model must get right (e.g. pedestrians at night),
and have confidence that they’re always shipping the best model.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from ..connection import Connection
from ..constants import NAME_KEY, SLICE_ID_KEY
from ..dataset_item import DatasetItem
from .constants import (
    EVAL_FUNCTION_ID_KEY,
    SCENARIO_TEST_ID_KEY,
    SCENARIO_TEST_METRICS_KEY,
    THRESHOLD_COMPARISON_KEY,
    THRESHOLD_KEY,
    ThresholdComparison,
)
from .data_transfer_objects.scenario_test_evaluations import GetEvalHistory
from .data_transfer_objects.scenario_test_metric import AddScenarioTestFunction
from .eval_functions.available_eval_functions import EvalFunction
from .scenario_test_evaluation import ScenarioTestEvaluation
from .scenario_test_metric import ScenarioTestMetric

DATASET_ITEMS_KEY = "dataset_items"


@dataclass
class ScenarioTest:
    """A Scenario Test combines a slice and at least one evaluation criterion. A :class:`ScenarioTest` is not created through
    the default constructor but using the instructions shown in :class:`Validate`. This :class:`ScenarioTest` class only
    simplifies the interaction with the scenario tests from this SDK.

    Attributes:
        id (str): The ID of the scenario test.
        connection (Connection): The connection to Nucleus API.
        name (str): The name of the scenario test.
        slice_id (str): The ID of the associated Nucleus slice.
    """

    id: str
    connection: Connection = field(repr=False)
    name: str = field(init=False)
    slice_id: str = field(init=False)
    baseline_model_id: Optional[str] = None

    def __post_init__(self):
        # TODO(gunnar): Remove this pattern. It's too slow. We should get all the info required in one call
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/info",
        )
        self.name = response[NAME_KEY]
        self.slice_id = response[SLICE_ID_KEY]

    def add_eval_function(
        self, eval_function: EvalFunction
    ) -> ScenarioTestMetric:
        """Creates and adds a new evaluation metric to the :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.create_scenario_test(
                "sample_scenario_test", "slc_bx86ea222a6g057x4380"
            )

            e = client.validate.eval_functions
            # Assuming a user would like to add all available public evaluation functions as criteria
            scenario_test.add_eval_function(
                e.bbox_iou
            )
            scenario_test.add_eval_function(
                e.bbox_map
            )
            scenario_test.add_eval_function(
                e.bbox_precision
            )
            scenario_test.add_eval_function(
                e.bbox_recall
            )

        Args:
            eval_function: :class:`EvalFunction`

        Returns:
            The created ScenarioTestMetric object.
        """
        response = self.connection.post(
            AddScenarioTestFunction(
                scenario_test_name=self.name,
                eval_function_id=eval_function.id,
            ).dict(),
            "validate/scenario_test_eval_function",
        )
        print(response)
        return ScenarioTestMetric(
            scenario_test_id=response[SCENARIO_TEST_ID_KEY],
            eval_function_id=response[EVAL_FUNCTION_ID_KEY],
            threshold=response.get(THRESHOLD_KEY, None),
            threshold_comparison=response.get(
                THRESHOLD_COMPARISON_KEY,
                ThresholdComparison.GREATER_THAN_EQUAL_TO,
            ),
            connection=self.connection,
        )

    def get_eval_functions(self) -> List[ScenarioTestMetric]:
        """Retrieves all criteria of the :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.scenario_tests[0]

            scenario_test.get_eval_functions()

        Returns:
            A list of ScenarioTestMetric objects.
        """
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/metrics",
        )
        return [
            ScenarioTestMetric(**metric, connection=self.connection)
            for metric in response[SCENARIO_TEST_METRICS_KEY]
        ]

    def get_eval_history(self) -> List[ScenarioTestEvaluation]:
        """Retrieves evaluation history for :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.scenario_tests[0]

            scenario_test.get_eval_history()

        Returns:
            A list of :class:`ScenarioTestEvaluation` objects.
        """
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/eval_history",
        )
        eval_history = GetEvalHistory.parse_obj(response)
        return [
            ScenarioTestEvaluation(evaluation.id, self.connection)
            for evaluation in eval_history.evaluations
        ]

    def get_items(self) -> List[DatasetItem]:
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/items",
        )
        return [
            DatasetItem.from_json(item) for item in response[DATASET_ITEMS_KEY]
        ]

    def set_baseline_model(self, model_id: str):
        """Set's a new baseline model for the ScenarioTest.  In order to be eligible to be a baseline,
        this scenario test must have been evaluated using that model.  The baseline model's performance
        is used as the threshold for all metrics against which other models are compared.

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.scenario_tests[0]

            scenario_test.set_baseline_model('my_baseline_model_id')

        Returns:
            A list of :class:`ScenarioTestEvaluation` objects.
        """
        response = self.connection.post(
            {},
            f"validate/scenario_test/{self.id}/set_baseline_model/{model_id}",
        )
        self.baseline_model_id = response.get("baseline_model_id")
        return self.baseline_model_id
