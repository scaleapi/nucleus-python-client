"""Unit Tests combine collections of data and evaluation metrics to accelerate model evaluation.

With Model CI Unit Tests, an ML engineer can define a Unit Test from critical
edge case scenarios that the model must get right (e.g. pedestrians at night),
and have confidence that they’re always shipping the best model.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List
import requests

from nucleus.connection import Connection
from nucleus.constants import NAME_KEY, SLICE_ID_KEY

from .unit_test_evaluation import UnitTestEvaluation
from .constants import (
    EVAL_FUNCTION_ID_KEY,
    ID_KEY,
    THRESHOLD_COMPARISON_KEY,
    THRESHOLD_KEY,
    UNIT_TEST_ID_KEY,
    UNIT_TEST_NAME_KEY,
)

EVALUATIONS_KEY = "evaluations"


class ThresholdComparison(Enum):
    """Comparator between the result and the threshold."""

    GREATER_THAN = "greater_than"
    GREATER_THAN_EQUAL_TO = "greater_than_equal_to"
    LESS_THAN = "less_than"
    LESS_THAN_EQUAL_TO = "less_than_equal_to"


@dataclass(frozen=True)
class UnitTestMetric:
    """A Unit Test Metric is an evaluation function and comparator associated with a Unit Test.
    Note that this class is immutable.

    Attributes:
        unit_test_id (str): The ID of the associated unit test.
        eval_function_id (str): The ID of the associated evaluation function.
        threshold (float): The threshold for the unit test.
        threshold_comparison (ThresholdComparison): The comparator for the unit test.
    """

    unit_test_id: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison


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
        response = self.connection.make_request(
            {},
            f"modelci/unit_test/{self.id}/info",
            requests_command=requests.get,
        )
        self.name = response[NAME_KEY]
        self.slice_id = response[SLICE_ID_KEY]

    def add_metric(
        self,
        eval_function_id: str,
        threshold: float,
        threshold_comparison: ThresholdComparison,
    ) -> UnitTestMetric:
        """Creates and adds a new metric to the Unit Test. ::

            import nucleus
            from nucleus.modelci.unit_test import ThresholdComparison
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            unit_test.add_metric(
                eval_function_id="ef_c61595wh49km7ppkk14g",
                threshold=0.5,
                threshold_comparison=ThresholdComparison.GREATER_THAN
            )

        Args:
            eval_function_id: ID of evaluation function
            threshold: numerical threshold that together with threshold comparison, defines success criteria for test evaluation.
            threshold_comparison: comparator for evaluation. i.e. threshold=0.5 and threshold_comparator > implies that a test only passes if score > 0.5.

        Returns:
            The created UnitTestMetric object.
        """
        response = self.connection.make_request(
            {
                UNIT_TEST_NAME_KEY: self.name,
                EVAL_FUNCTION_ID_KEY: eval_function_id,
                THRESHOLD_KEY: threshold,
                THRESHOLD_COMPARISON_KEY: threshold_comparison.value,
            },
            "modelci/unit_test_metric",
            requests_command=requests.post,
        )
        return UnitTestMetric(
            unit_test_id=response[UNIT_TEST_ID_KEY],
            eval_function_id=response[EVAL_FUNCTION_ID_KEY],
            threshold=threshold,
            threshold_comparison=threshold_comparison,
        )

    def get_metrics(self) -> List[UnitTestMetric]:
        """Retrieves all metrics of the Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            unit_test.get_metrics()

        Returns:
            A list of UnitTestMetric objects.
        """
        response = self.connection.make_request(
            {},
            f"modelci/unit_test/{self.id}/metrics",
            requests_command=requests.get,
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
        response = self.connection.make_request(
            {},
            f"modelci/unit_test/{self.id}/eval_history",
            requests_command=requests.get,
        )
        return [
            UnitTestEvaluation(eval[ID_KEY], self.connection)
            for eval in response[EVALUATIONS_KEY]
        ]
