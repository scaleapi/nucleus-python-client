from nucleus.pydantic_base import ImmutableModel

from .constants import ThresholdComparison


class ScenarioTestMetric(ImmutableModel):
    """A Scenario Test Metric is an evaluation function combined with a comparator and associated with a Scenario Test.
    Scenario Test Metrics serve as the basis when evaluating a Model on a Scenario Test.
    """

    scenario_test_id: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison
