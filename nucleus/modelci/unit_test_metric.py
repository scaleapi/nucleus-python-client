from nucleus.pydantic_base import ImmutableModel

from .constants import ThresholdComparison


class UnitTestMetric(ImmutableModel):
    """A Unit Test Metric is an evaluation function combined with a comparator and associated with a Unit Test.
    Unit Test Metrics serve as the basis when evaluating a Model on a Unit Test.
    """

    unit_test_id: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison
