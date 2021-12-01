"""Unit Tests combine collections of data and evaluation metrics to accelerate model evaluation.

With Model CI Unit Tests, an ML engineer can define a Unit Test from critical
edge case scenarios that the model must get right (e.g. pedestrians at night),
and have confidence that they’re always shipping the best model.
"""
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from .data_transfer_objects.add_unit_test_metric import AddUnitTestMetric
from .unit_test_evaluation import UnitTestEvaluation
from .unit_test_metric import UnitTestMetric
from .constants import ThresholdComparison


@dataclass
class UnitTestInfo:
    name: str
    slice_id: str


if TYPE_CHECKING:
    from nucleus.modelci import (
        ModelCI,
        UnitTestMetric,
        UNIT_TEST_ID_KEY,
        EVAL_FUNCTION_ID_KEY,
        UNIT_TEST_NAME_KEY,
        THRESHOLD_KEY,
        THRESHOLD_COMPARISON_KEY,
        EvalFunctionCondition,
    )


class UnitTest:
    """A Unit Test combines a slice and at least one evaluation metric."""

    def __init__(
        self,
        unit_test_id: str,
        client: "ModelCI",  # type:ignore # noqa: F821
    ):
        self.id = unit_test_id
        self._client: ModelCI = client
        info = self.info()
        self.name = info.name
        self.slice_id = info.slice_id

    def __repr__(self):
        return f"UnitTest(name='{self.name}', slice_id='{self.slice_id}', unit_test_id='{self.id}')"

    def __eq__(self, other):
        return self.id == other.id and self._client == other._client

    def add_metric(
        self, evaluation_condition: EvalFunctionCondition
    ) -> UnitTestMetric:
        """Creates and adds a new metric to the Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            iou = client.modelci.eval_functions.iou
            unit_test.add_metric(
                iou() > 0.5
            )

        Args:
            evaluation_condition: :class:`EvalFunctionCondition` created by comparison with an :class:`EvalFunction`

        Returns:
            The created UnitTestMetric object.
        """
        response = self._client._connection.post(
            AddUnitTestMetric(
                unit_test_name=self.name,
                eval_function_id=evaluation_condition.eval_function_id,
                threshold=evaluation_condition.threshold,
                threshold_comparison=evaluation_condition.threshold_comparison,
            ).dict(),
            "modelci/unit_test_metric",
        )
        return UnitTestMetric(
            unit_test_id=response[UNIT_TEST_ID_KEY],
            eval_function_id=response[EVAL_FUNCTION_ID_KEY],
            threshold=evaluation_condition.threshold,
            threshold_comparison=evaluation_condition.threshold_comparison,
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
        response = self._client._connection.get(
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
        return self._client.get_unit_test_eval_history(self.id)

    def info(self):
        """Retrieves info of the Unit Test.

        Args:
            unit_test_id: ID of Unit Test

        Returns:
            A UnitTestInfo object
        """
        response = self._client._connection.get(
            f"modelci/unit_test/{self.id}/info",
        )
        return UnitTestInfo(**response)
