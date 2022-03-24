from dataclasses import field
from typing import Optional

from nucleus.pydantic_base import ImmutableModel

from ..connection import Connection
from .constants import ThresholdComparison


class ScenarioTestMetric(ImmutableModel):
    """A Scenario Test Metric is an evaluation function combined with a comparator and associated with a Scenario Test.
    Scenario Test Metrics serve as the basis when evaluating a Model on a Scenario Test.
    """

    scenario_test_id: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison
    connection: Connection = field(repr=False)

    def set_threshold(self, threshold: Optional[float] = None) -> None:
        """Sets the threshold of the metric to the new value passed in as a parameters.

        Attributes:
            threshold (str): The ID of the scenario test.
        """
        response = self.connection.post(
            f"/nucleus/validate/metric/set_threshold/{self.scenario_test_id}/{self.eval_function_id}"
        )
        self.threshold = response.threshold
