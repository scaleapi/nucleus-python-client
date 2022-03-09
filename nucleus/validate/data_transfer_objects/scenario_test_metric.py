from nucleus.pydantic_base import ImmutableModel

from ..constants import ThresholdComparison


class AddScenarioTestMetric(ImmutableModel):
    """Data transfer object to add a scenario test."""

    scenario_test_name: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison
