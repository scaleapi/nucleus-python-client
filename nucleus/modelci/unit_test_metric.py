from dataclasses import dataclass
from enum import Enum


@dataclass
class ThresholdComparison(str, Enum):
    """Comparator between the result and the threshold."""

    GREATER_THAN = "greater_than"
    GREATER_THAN_EQUAL_TO = "greater_than_equal_to"
    LESS_THAN = "less_than"
    LESS_THAN_EQUAL_TO = "less_than_equal_to"

@dataclass
class UnitTestMetric:
    # TODO(gunnar): This seems like a condition -> UnitTestCondition? UnitTestAssertion?
    """A Unit Test Metric is an evaluation function and comparator associated with a Unit Test."""

    unit_test_id: str
    eval_function_id: str
    threshold: float
    threshold_comparison: ThresholdComparison