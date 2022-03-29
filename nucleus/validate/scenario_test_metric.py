from dataclasses import dataclass, field
from typing import Dict, Optional

from ..connection import Connection
from .constants import ThresholdComparison


@dataclass
class ScenarioTestMetric:
    """A Scenario Test Metric is an evaluation function combined with a comparator and associated with a Scenario Test.
    Scenario Test Metrics serve as the basis when evaluating a Model on a Scenario Test.
    """

    scenario_test_id: str
    eval_function_id: str
    threshold: Optional[float]
    connection: Connection
    eval_func_arguments: Optional[Dict] = field(default_factory=dict)
    threshold_comparison: ThresholdComparison = (
        ThresholdComparison.GREATER_THAN_EQUAL_TO
    )

    def set_threshold(self, threshold: Optional[float] = None) -> None:
        """Sets the threshold of the metric to the new value passed in as a parameters.
        Attributes:
            threshold (str): The ID of the scenario test.
        """
        payload = {"threshold": threshold}
        response = self.connection.post(
            payload,
            f"validate/metric/set_threshold/{self.scenario_test_id}/{self.eval_function_id}",
        )
        self.threshold = response.get("threshold")
