"""Scenario Tests combine collections of data and evaluation metrics to accelerate model evaluation.

With Model CI Scenario Tests, an ML engineer can define a Scenario Test from critical
edge case scenarios that the model must get right (e.g. pedestrians at night),
and have confidence that they’re always shipping the best model.
"""
from dataclasses import dataclass, field
from typing import List

from ..connection import Connection
from ..constants import NAME_KEY, SLICE_ID_KEY
from ..dataset_item import DatasetItem
from .data_transfer_objects.eval_function import EvaluationCriterion
from .data_transfer_objects.scenario_test_evaluations import GetEvalHistory
from .data_transfer_objects.scenario_test_metric import AddScenarioTestMetric
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

    def __post_init__(self):
        # TODO(gunnar): Remove this pattern. It's too slow. We should get all the info required in one call
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/info",
        )
        self.name = response[NAME_KEY]
        self.slice_id = response[SLICE_ID_KEY]

    def add_criterion(
        self, evaluation_criterion: EvaluationCriterion
    ) -> ScenarioTestMetric:
        """Creates and adds a new criteria to the :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.create_scenario_test(
                "sample_scenario_test", "slc_bx86ea222a6g057x4380"
            )

            e = client.validate.eval_functions
            # Assuming a user would like to add all available public evaluation functions as criteria
            scenario_test.add_criterion(
                e.bbox_iou() > 0.5
            )
            scenario_test.add_criterion(
                e.bbox_map() > 0.85
            )
            scenario_test.add_criterion(
                e.bbox_precision() > 0.7
            )
            scenario_test.add_criterion(
                e.bbox_recall() > 0.6
            )

        Args:
            evaluation_criterion: :class:`EvaluationCriterion` created by comparison with an :class:`EvalFunction`

        Returns:
            The created ScenarioTestMetric object.
        """
        response = self.connection.post(
            AddScenarioTestMetric(
                scenario_test_name=self.name,
                eval_function_id=evaluation_criterion.eval_function_id,
                threshold=evaluation_criterion.threshold,
                threshold_comparison=evaluation_criterion.threshold_comparison,
            ).dict(),
            "validate/scenario_test_metric",
        )
        return ScenarioTestMetric(
            scenario_test_id=response["scenario_test_id"],
            eval_function_id=response["eval_function_id"],
            threshold=evaluation_criterion.threshold,
            threshold_comparison=evaluation_criterion.threshold_comparison,
        )

    def get_criteria(self) -> List[ScenarioTestMetric]:
        """Retrieves all criteria of the :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.list_scenario_tests()[0]

            scenario_test.get_criteria()

        Returns:
            A list of ScenarioTestMetric objects.
        """
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/metrics",
        )
        return [
            ScenarioTestMetric(**metric)
            for metric in response["scenario_test_metrics"]
        ]

    def get_eval_history(self) -> List[ScenarioTestEvaluation]:
        """Retrieves evaluation history for :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.list_scenario_tests()[0]

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
