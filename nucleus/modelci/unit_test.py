"""Unit Tests combine collections of data and evaluation metrics to accelerate model evaluation.

With Model CI Unit Tests, an ML engineer can define a Unit Test from critical
edge case scenarios that the model must get right (e.g. pedestrians at night),
and have confidence that they’re always shipping the best model.
"""
from dataclasses import dataclass, field
from typing import List

from ..connection import Connection
from ..constants import NAME_KEY, SLICE_ID_KEY
from ..dataset_item import DatasetItem
from .data_transfer_objects.eval_function import EvaluationCriterion
from .data_transfer_objects.unit_test_evaluations import GetEvalHistory
from .data_transfer_objects.unit_test_metric import AddUnitTestMetric
from .unit_test_evaluation import UnitTestEvaluation
from .unit_test_metric import UnitTestMetric

DATASET_ITEMS_KEY = "dataset_items"


@dataclass
class UnitTest:
    """A Unit Test combines a slice and at least one evaluation criterion. A :class:`UnitTest` is not created through
    the default constructor but using the instructions shown in :class:`ModelCI`. This :class:`UnitTest` class only
    simplifies the interaction with the unit tests from this SDK.

    Attributes:
        id (str): The ID of the unit test.
        connection (Connection): The connection to Nucleus API.
        name (str): The name of the unit test.
        slice_id (str): The ID of the associated Nucleus slice.
    """

    id: str
    connection: Connection = field(repr=False)
    name: str = field(init=False)
    slice_id: str = field(init=False)

    def __post_init__(self):
        response = self.connection.get(
            f"modelci/unit_test/{self.id}/info",
        )
        self.name = response[NAME_KEY]
        self.slice_id = response[SLICE_ID_KEY]

    def add_criterion(
        self, evaluation_criterion: EvaluationCriterion
    ) -> UnitTestMetric:
        """Creates and adds a new criteria to the :class:`UnitTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            e = client.modelci.eval_functions
            # Assuming a user would like to add all available public evaluation functions as criteria
            unit_test.add_criterion(
                e.bbox_iou() > 0.5
            )
            unit_test.add_criterion(
                e.bbox_map() > 0.85
            )
            unit_test.add_criterion(
                e.bbox_precision() > 0.7
            )
            unit_test.add_criterion(
                e.bbox_recall() > 0.6
            )

        Args:
            evaluation_criterion: :class:`EvaluationCriterion` created by comparison with an :class:`EvalFunction`

        Returns:
            The created UnitTestMetric object.
        """
        response = self.connection.post(
            AddUnitTestMetric(
                unit_test_name=self.name,
                eval_function_id=evaluation_criterion.eval_function_id,
                threshold=evaluation_criterion.threshold,
                threshold_comparison=evaluation_criterion.threshold_comparison,
            ).dict(),
            "modelci/unit_test_metric",
        )
        return UnitTestMetric(
            unit_test_id=response["unit_test_id"],
            eval_function_id=response["eval_function_id"],
            threshold=evaluation_criterion.threshold,
            threshold_comparison=evaluation_criterion.threshold_comparison,
        )

    def get_criteria(self) -> List[UnitTestMetric]:
        """Retrieves all criteria of the :class:`UnitTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            unit_test.get_criteria()

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
        """Retrieves evaluation history for :class:`UnitTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            unit_test.get_eval_history()

        Returns:
            A list of :class:`UnitTestEvaluation` objects.
        """
        response = self.connection.get(
            f"modelci/unit_test/{self.id}/eval_history",
        )
        eval_history = GetEvalHistory.parse_obj(response)
        return [
            UnitTestEvaluation(evaluation.id, self.connection)
            for evaluation in eval_history.evaluations
        ]

    def get_items(self) -> List[DatasetItem]:
        response = self.connection.get(
            f"modelci/unit_test/{self.id}/items",
        )
        return [
            DatasetItem.from_json(item) for item in response[DATASET_ITEMS_KEY]
        ]
