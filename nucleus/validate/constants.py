from enum import Enum

EVAL_FUNCTION_KEY = "eval_fn"
EVALUATION_ID_KEY = "evaluation_id"
EVAL_FUNCTION_ID_KEY = "eval_function_id"
ID_KEY = "id"
PASS_KEY = "pass"
RESULT_KEY = "result"
THRESHOLD_COMPARISON_KEY = "threshold_comparison"
THRESHOLD_KEY = "threshold"
SCENARIO_TEST_ID_KEY = "scenario_test_id"
SCENARIO_TEST_NAME_KEY = "scenario_test_name"
SCENARIO_TEST_METRICS_KEY = "scenario_test_metrics"


class ThresholdComparison(str, Enum):
    """Comparator between the result and the threshold."""

    GREATER_THAN = "greater_than"
    GREATER_THAN_EQUAL_TO = "greater_than_equal_to"
    LESS_THAN = "less_than"
    LESS_THAN_EQUAL_TO = "less_than_equal_to"


class EntityLevel(str, Enum):
    """
    Data level at which evaluation functions produce outputs.
    For instance, when comparing results across dataset items, use
    `EntityLevel.ITEM`. For scenes, use `EntityLevel.SCENE`. Finally,
    when comparing results between tracks within a single scene or a
    holistic item datset, use `EntityLevel.TRACK`.
    """

    TRACK = "track"
    ITEM = "item"
    SCENE = "scene"
