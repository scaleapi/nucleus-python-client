"""Unit Tests combine collections of data and evaluation metrics to accelerate model evaluation.

With Model CI Unit Tests, an ML engineer can define a Unit Test from critical
edge case scenarios that the model must get right (e.g. pedestrians at night),
and have confidence that they’re always shipping the best model.
"""
from dataclasses import dataclass, field
from typing import List

from ..connection import Connection
from ..constants import NAME_KEY, SLICE_ID_KEY
from .data_transfer_objects.eval_function import EvaluationCriterion
from .data_transfer_objects.unit_test_evaluations import GetEvalHistory
from .data_transfer_objects.unit_test_metric import AddUnitTestMetric
from .unit_test_evaluation import UnitTestEvaluation
from .unit_test_metric import UnitTestMetric


@dataclass
class UnitTest:
    """A Unit Test combines a slice and at least one evaluation metric.

    Attributes:
        id (str): The ID of the unit test.
        connection (Connection): The connection to Nucleus API.
        name (str): The name of the unit test.
        slice_id (str): The ID of the associated Nucleus slice.
    """

    id: str
    connection: Connection
    name: str = field(init=False)
    slice_id: str = field(init=False)

    def __post_init__(self):
        response = self.connection.get(
            f"modelci/unit_test/{self.id}/info",
        )
        self.name = response[NAME_KEY]
        self.slice_id = response[SLICE_ID_KEY]

    def add_criteria(
        self, evaluation_criteria: EvaluationCriterion
    ) -> UnitTestMetric:
        """Creates and adds a new criteria to the Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            iou = client.modelci.eval_functions.bbox_iou
            unit_test.add_metric(
                iou() > 0.5
            )

        Args:
            evaluation_criteria: :class:`EvaluationCriterion` created by comparison with an :class:`EvalFunction`

        Returns:
            The created UnitTestMetric object.
        """
        response = self.connection.post(
            AddUnitTestMetric(
                unit_test_name=self.name,
                eval_function_id=evaluation_criteria.eval_function_id,
                threshold=evaluation_criteria.threshold,
                threshold_comparison=evaluation_criteria.threshold_comparison,
            ).dict(),
            "modelci/unit_test_metric",
        )
        return UnitTestMetric(
            unit_test_id=response["unit_test_id"],
            eval_function_id=response["eval_function_id"],
            threshold=evaluation_criteria.threshold,
            threshold_comparison=evaluation_criteria.threshold_comparison,
        )

    def get_criteria(self) -> List[UnitTestMetric]:
        """Retrieves all metrics of the Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            unit_test.get_metrics()

        Returns:
            A list of UnitTestMetric objects.
        """
        response = self.connection.get(
            f"modelci/unit_test/{self.id}/metrics",
        )
        return [
            UnitTestMetric(**metric)
            for metric in response["unit_test_metrics"]
        ]

    def get_eval_history(self) -> List[UnitTestEvaluation]:
        """Retrieves evaluation history for Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            unit_test.get_eval_history()

        Returns:
            A list of UnitTestEvaluation objects.
        """
        response = self.connection.get(
            f"modelci/unit_test/{self.id}/eval_history",
        )
        # TODO(gunnar): Repeated info calls are slow -> Move work to backend
        eval_history = GetEvalHistory.parse_obj(response)
        return [
            self.get_unit_test_eval_info(evaluation.id)
            for evaluation in eval_history.evaluations
        ]

    def get_unit_test_eval_info(
        self, evaluation_id: str
    ) -> UnitTestEvaluation:
        """Retrieves info of the Unit Test Evaluation.

        Args:
            evaluation_id: ID of Unit Test

        Returns:
            A list of UnitTestEvaluation objects
        """
        # TODO(gunnar): Use pydantic and remove need to call info under the hood
        return UnitTestEvaluation(id=evaluation_id, connection=self.connection)
