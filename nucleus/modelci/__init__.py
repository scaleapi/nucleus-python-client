"""Model CI Python Library."""
from typing import List

import requests

from nucleus.job import AsyncJob
from nucleus.constants import (
    NAME_KEY,
    SLICE_ID_KEY,
)
from nucleus.connection import Connection

from .constants import (
    EVAL_FUNCTION_ID_KEY,
    ID_KEY,
    THRESHOLD_COMPARISON_KEY,
    THRESHOLD_KEY,
    UNIT_TEST_ID_KEY,
    UNIT_TEST_NAME_KEY,
)
from .eval_function import EvalFunction
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

SUCCESS_KEY = "success"
EVALUATIONS_KEY = "evaluations"
EVAL_FUNCTIONS_KEY = "eval_functions"


class ModelCI:
    """Model CI Python Client extension."""

    def __init__(self, api_key: str, endpoint: str):
        self._connection = Connection(api_key, endpoint)

    def __repr__(self):
        return f"ModelCI(connection='{self._connection}')"

    def __eq__(self, other):
        return self._connection == other._connection

    def list_eval_functions(self) -> List[EvalFunction]:
        """List all available evaluation functions. ::

        import nucleus
        client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

        eval_functions = client.modelci.list_eval_functions()
        """
        response = self._connection.make_request(
            {},
            "modelci/eval_fn",
            requests_command=requests.get,
        )
        return [
            EvalFunction(**eval_function)
            for eval_function in response[EVAL_FUNCTIONS_KEY]
        ]

    def create_unit_test(self, name: str, slice_id: str) -> UnitTest:
        """Creates a new Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

        Args:
            name: unique name of test
            slice_id: id of slice of items to evaluate test on.

        Returns:
            Created UnitTest object.
        """
        response = self._connection.make_request(
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
        eval_function_id: str,
        threshold: float,
        threshold_comparison: ThresholdComparison,
    ) -> UnitTestMetric:
        """Creates and adds a new metric for the provided Unit Test. ::

            import nucleus
            from nucleus.modelci.unit_test import ThresholdComparison
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            client.modelci.create_unit_test_metric(
                unit_test_name="sample_unit_test",
                eval_function_id="ef_c61595wh49km7ppkk14g",
                threshold=0.5,
                threshold_comparison=ThresholdComparison.GREATER_THAN
            )

        Args:
            unit_test_name: name of unit test
            eval_function_id: name of evaluation function
            threshold: numerical threshold that together with threshold comparison, defines success criteria for test evaluation.
            threshold_comparison: comparator for evaluation. i.e. threshold=0.5 and threshold_comparator > implies that a test only passes if score > 0.5.

        Returns:
            The created UnitTestMetric object.
        """
        response = self._connection.make_request(
            {
                UNIT_TEST_NAME_KEY: unit_test_name,
                EVAL_FUNCTION_ID_KEY: eval_function_id,
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
        """Retrieves info of the Unit Test.

        Args:
            unit_test_id: ID of Unit Test

        Returns:
            A UnitTestInfo object
        """
        response = self._connection.make_request(
            {},
            f"modelci/unit_test/{unit_test_id}/info",
            requests_command=requests.get,
        )
        return UnitTestInfo(**response)

    def list_unit_tests(self) -> List[UnitTest]:
        """Lists all Unit Tests of the current user. ::

            import nucleus
            from nucleus.modelci.unit_test import ThresholdComparison
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            client.modelci.list_unit_tests()

        Returns:
            A list of UnitTest objects.
        """
        response = self._connection.make_request(
            {},
            "modelci/unit_test",
            requests_command=requests.get,
        )
        return [
            UnitTest(test_id, self) for test_id in response["unit_test_ids"]
        ]

    def get_unit_test_metrics(self, unit_test_id: str) -> List[UnitTestMetric]:
        """Retrieves all metrics of the Unit Test. ::

            import nucleus.modelci as nm
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            client.modelci.get_unit_test_metrics(unit_test.id)

        Args:
            unit_test_id: ID of Unit Test

        Returns:
            A list of UnitTestMetric objects
        """
        response = self._connection.make_request(
            {},
            f"modelci/unit_test/{unit_test_id}/metrics",
            requests_command=requests.get,
        )
        return [
            UnitTestMetric(**metric)
            for metric in response["unit_test_metrics"]
        ]

    def delete_unit_test(self, unit_test_id: str) -> bool:
        """Creates a new Unit Test. ::

            import nucleus
            from nucleus.modelci.unit_test import ThresholdComparison
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            success = client.modelci.create_unit_test(unit_test.id)

        Args:
            unit_test_id: unique ID of unit test

        Returns:
            Whether deletion was successful.
        """
        response = self._connection.make_request(
            {},
            f"modelci/unit_test/{unit_test_id}",
            requests_command=requests.delete,
        )
        return response[SUCCESS_KEY]

    def evaluate_model_on_unit_tests(
        self, model_id: str, unit_test_names: List[str]
    ) -> AsyncJob:
        """Evaluates the given model on the specified Unit Tests. ::

            import nucleus.modelci as nm
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            client.modelci.evaluate_model_on_unit_tests(
                model_id=model.id,
                unit_test_names=["sample_unit_test"],
            )

        Args:
            model_id: ID of model to evaluate
            unit_test_names: list of unit tests to evaluate

        Returns:
            AsyncJob object of evaluation job
        """
        response = self._connection.make_request(
            {"test_names": unit_test_names},
            f"modelci/{model_id}/evaluate",
            requests_command=requests.post,
        )
        return AsyncJob.from_json(response, self._connection)

    def get_unit_test_eval_history(
        self, unit_test_id: str
    ) -> List[UnitTestEvaluation]:
        """Retrieves evaluation history of the Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            client.modelci.get_unit_test_eval_history(unit_test.id)

        Args:
            unit_test_id: ID of Unit Test

        Returns:
            A list of UnitTestEvaluation objects
        """
        response = self._connection.make_request(
            {},
            f"modelci/unit_test/{unit_test_id}/eval_history",
            requests_command=requests.get,
        )
        return [
            self.get_unit_test_eval_info(eval[ID_KEY])
            for eval in response[EVALUATIONS_KEY]
        ]

    def get_unit_test_eval_info(
        self, evaluation_id: str
    ) -> UnitTestEvaluation:
        """Retrieves info of the Unit Test Evaluation.

        Args:
            evaluation_id: ID of Unit Test

        Returns:
            A list of UnitTestEvaluation objects
        """
        response = self._connection.make_request(
            {},
            f"modelci/eval/{evaluation_id}/info",
            requests_command=requests.get,
        )
        return format_unit_test_eval_response(response)
