from enum import Enum

EVAL_FUNCTION_ID_KEY = "eval_function_id"
ID_KEY = "id"
THRESHOLD_COMPARISON_KEY = "threshold_comparison"
THRESHOLD_KEY = "threshold"
UNIT_TEST_ID_KEY = "unit_test_id"
UNIT_TEST_NAME_KEY = "unit_test_name"


class ThresholdComparison(str, Enum):
    """Comparator between the result and the threshold."""

    GREATER_THAN = "greater_than"
    GREATER_THAN_EQUAL_TO = "greater_than_equal_to"
    LESS_THAN = "less_than"
    LESS_THAN_EQUAL_TO = "less_than_equal_to"