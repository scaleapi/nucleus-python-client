"""Unit Tests combine collections of data and evaluation metrics to accelerate model evaluation.

With Model CI Unit Tests, an ML engineer can define a Unit Test from critical
edge case scenarios that the model must get right (e.g. pedestrians at night),
and have confidence that they’re always shipping the best model.
"""
from enum import Enum
from dataclasses import dataclass
from typing import List

from .unit_test_evaluation import UnitTestEvaluation


@dataclass
class ThresholdComparison(str, Enum):
    """Comparator between the result and the threshold."""

    GREATER_THAN = "greater_than"
    GREATER_THAN_EQUAL_TO = "greater_than_equal_to"
    LESS_THAN = "less_than"
    LESS_THAN_EQUAL_TO = "less_than_equal_to"


@dataclass
class UnitTestInfo:
    name: str
    slice_id: str


@dataclass
class UnitTestMetric:
    """A Unit Test Metric is an evaluation function and comparator associated with a Unit Test."""

    unit_test_id: str
    eval_function_id: str
    threshold: float
    threshold_comparison: str


class UnitTest:
    """A Unit Test combines a slice and at least one evaluation metric."""

    def __init__(
        self,
        unit_test_id: str,
        client: "ModelCI",  # type:ignore # noqa: F821
    ):
        # Hack: lazy import to avoid circular dependencies.
        from nucleus.modelci import ModelCI

        self.id = unit_test_id
        self._client: ModelCI = client
        info = self._client.get_unit_test_info(self.id)
        self.name = info.name
        self.slice_id = info.slice_id

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
        return self._client.create_unit_test_metric(
            self.name, eval_function_id, threshold, threshold_comparison
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
        return self._client.get_unit_test_metrics(self.id)

    def get_eval_history(self) -> List[UnitTestEvaluation]:
        """Retrieves evaluation history for Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            unit_test.get_eval_history()

        Returns:
            A list of UnitTestEvaluation objects.
        """
        return self._client.get_unit_test_eval_history(self.id)
