from dataclasses import dataclass

from nucleus.modelci import ThresholdComparison


@dataclass
class UnitTestMetric:
    # TODO(gunnar): This seems like a condition -> UnitTestCondition? UnitTestAssertion?
    """A Unit Test Metric is an evaluation function and comparator associated with a Unit Test."""

    unit_test_id: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison
