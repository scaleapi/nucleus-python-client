from enum import Enum
from dataclasses import dataclass

from nucleus import NucleusClient
from nucleus.constants import NAME_KEY, SLICE_ID_KEY


@dataclass
class ThresholdComparison(str, Enum):
    GREATER_THAN = ("greater_than",)
    GREATER_THAN_EQUAL_TO = ("greater_than_equal_to",)
    LESS_THAN = ("less_than",)
    LESS_THAN_EQUAL_TO = ("less_than_equal_to",)


@dataclass
class UnitTestInfo():
    name: str
    slice_id: str


@dataclass
class UnitTestMetric():
    unit_test_id: str
    eval_function_id: str
    threshold: float
    threshold_comparison: str


class UnitTest:
    """
    Model CI Unit Test.
    """

    def __init__(self, unit_test_id: str, client: NucleusClient):
        self.id = unit_test_id
        self._client = client
        info = self._client.get_unit_test_info(self.id)
        self.name = info[NAME_KEY]
        self.slice_id = info[SLICE_ID_KEY]

    def get_eval_history(self):
        """
        Get evaluation history for unit test.
        :return: List[UnitTestEvaluation]
        """
        return self._client.get_unit_test_eval_history(self.id)

    def get_metrics(self):
        """
        Get metrics for the unit test.
        :return: List[UnitTestMetric]
        """
        return self._client.get_unit_test_metrics(self.id)

    def add_metric(
        self,
        eval_function_name: str,
        threshold: float,
        threshold_comparison: ThresholdComparison,
    ):
        """
        Create a modelCI unit test metric.  Takes an evaluation function name,
        evaluation threshold, and comparator as input.

        :param
        eval_function_name: name of evaluation function
        :param
        threshold: numerical threshold that together with threshold comparison,
        defines success criteria for test evaluation.
        :param
        threshold_comparison: comparator for evaluation. i.e. threshold=0.5
        and threshold_comparator > implies that a test only passes if
        score > 0.5.
        :return: UnitTestMetric typed dict
        """
        return self._client.create_unit_test_metric(
            self.name, eval_function_name, threshold, threshold_comparison
        )
