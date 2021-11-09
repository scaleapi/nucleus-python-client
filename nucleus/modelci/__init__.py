"""
Model CI Python Library.
"""
from typing import List

import requests

from nucleus import NucleusClient
from nucleus.job import AsyncJob
from nucleus.constants import (
    ID_KEY,
    NAME_KEY,
    SLICE_ID_KEY,
)

from .constants import (
    EVAL_FUNCTION_ID_KEY,
    EVAL_FUNCTION_NAME_KEY,
    THRESHOLD_COMPARISON_KEY,
    THRESHOLD_KEY,
    UNIT_TEST_ID_KEY,
    UNIT_TEST_NAME_KEY,
)
from .unit_test import (
    ThresholdComparison,
    UnitTest,
    UnitTestInfo,
    UnitTestMetric,
)
from .unit_test_evaluation import (
    UnitTestEvaluation,
    UnitTestItemEvaluation,
)
from .utils import format_unit_test_eval_response


class ModelCIClient(NucleusClient):
    """
    Model CI client.
    """

    def __init__(
        self,
        api_key: str,
        use_notebook: bool = False,
        endpoint: str = None,
    ):
        super().__init__(api_key, use_notebook, endpoint)

    def create_unit_test(self, name: str, slice_id: str) -> UnitTest:
        """
        Create a modelCI unit test.  Takes a test name and slice ID.

        :param
        name: unique name of test
        :param
        slice_id: id of slice of items to evaluate test on.
        :return: a UnitTest object
        """
        response = self.make_request(
            {
                NAME_KEY: name,
                SLICE_ID_KEY: slice_id,
            },
            "modelci/unit_test",
            requests_command=requests.post,
        )
        return UnitTest(response[UNIT_TEST_ID_KEY], self)

    def create_unit_test_metric(
        self,
        unit_test_name: str,
        eval_function_name: str,
        threshold: float,
        threshold_comparison: ThresholdComparison,
    ) -> UnitTestMetric:
        """
        Create a modelCI unit test metric.  Takes a unit test name, evaluation
        function name, evaluation threshold, and comparator as input.

        :param
        unit_test_name: name of unit test
        :param
        eval_function_name: name of evaluation function
        :param
        threshold: numerical threshold that together with threshold comparison,
        defines success criteria for test evaluation.
        :param
        threshold_comparison: comparator for evaluation. i.e. threshold=0.5
        and threshold_comparator > implies that a test only passes if
        score > 0.5.
        :return: a UnitTestMetric typed dict.
        """
        response = self.make_request(
            {
                UNIT_TEST_NAME_KEY: unit_test_name,
                EVAL_FUNCTION_NAME_KEY: eval_function_name,
                THRESHOLD_KEY: threshold,
                THRESHOLD_COMPARISON_KEY: threshold_comparison,
            },
            "modelci/unit_test_metric",
            requests_command=requests.post,
        )
        return UnitTestMetric(
            unit_test_id=response[UNIT_TEST_ID_KEY],
            eval_function_id=response[EVAL_FUNCTION_ID_KEY],
            threshold=threshold,
            threshold_comparison=threshold_comparison,
        )

    def get_unit_test_info(self, unit_test_id: str) -> UnitTestInfo:
        """
        Get unit test info. Takes in unit test ID.
        :param
        unit_test_id: ID of unit test
        :return: a UnitTestInfo object
        """
        response = self.make_request(
            {},
            f"modelci/unit_test/{unit_test_id}/info",
            requests_command=requests.get,
        )
        return UnitTestInfo(**response)

    def get_unit_test_eval_history(
        self, unit_test_id: str
    ) -> List[UnitTestEvaluation]:
        """
        Get evaluation history for unit test. Takes in unit test ID.
        :param
        unit_test_id: ID of unit test
        :return: List[UnitTestEvaluation]
        """
        response = self.make_request(
            {},
            f"modelci/unit_test/{unit_test_id}/eval_history",
            requests_command=requests.get,
        )
        return [
            UnitTestEvaluation(eval[ID_KEY])
            for eval in response["evaluations"]
        ]

    def get_unit_test_metrics(self, unit_test_id: str) -> List[UnitTestMetric]:
        """
        Get metrics for the unit test.
        :return: List[UnitTestMetric]
        """
        response = self.make_request(
            {},
            f"modelci/unit_test/{unit_test_id}/metrics",
            requests_command=requests.get,
        )
        return [UnitTestMetric(**metric) for metric in response["metrics"]]

    def get_unit_test_eval_info(
        self, evaluation_id: str
    ) -> UnitTestEvaluation:
        """
        Get info for a given unit test evaluation ID.
        :param
        evaluation_id: ID of unit test evaluation
        :return: List[UnitTestEvaluation]
        """
        response = self.make_request(
            {},
            f"modelci/eval/{evaluation_id}/info",
            requests_command=requests.get,
        )
        return format_unit_test_eval_response(response)

    def evaluate_model_on_unit_tests(
        self, model_id: str, unit_test_names: List[str]
    ) -> AsyncJob:
        """
        Evaluates the given model on the provided unit tests.
        :param
        model_id: ID of model to evaluate
        :param
        unit_test_names: names of unit tests to evaluate
        :return: AsyncJob object
        """

        response = self.make_request(
            {"test_names": unit_test_names},
            f"modelci/{model_id}/evaluate",
            requests_command=requests.post,
        )
        return AsyncJob.from_json(response, self._client)
