from nucleus.pydantic_base import ImmutableModel

from ..constants import ThresholdComparison


class AddUnitTestMetric(ImmutableModel):
    """ Data transfer object to add a unit test."""

    unit_test_name: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison
