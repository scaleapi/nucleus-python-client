"""Model CI Python Library."""
from typing import List

import requests

from nucleus.connection import Connection
from nucleus.constants import NAME_KEY, SLICE_ID_KEY
from nucleus.job import AsyncJob

from .constants import UNIT_TEST_ID_KEY
from .eval_function import EvalFunction
from .unit_test import UnitTest, UnitTestMetric

SUCCESS_KEY = "success"
EVAL_FUNCTIONS_KEY = "eval_functions"


class ModelCI:
    """Model CI Python Client extension."""

    def __init__(self, api_key: str, endpoint: str):
        self._connection = Connection(api_key, endpoint)

    def __repr__(self):
        return f"ModelCI(connection='{self._connection}')"

    def __eq__(self, other):
        return self._connection == other.connection

    def list_eval_functions(self) -> List[EvalFunction]:
        """Lists all available evaluation functions. ::

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
        """Creates a new Unit Test from an existing Slice within Nucleus. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

        Args:
            name: unique name of test
            slice_id: id of a (pre-defined) slice of items to evaluate test on.

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
        return UnitTest(response[UNIT_TEST_ID_KEY], self._connection)

    def list_unit_tests(self) -> List[UnitTest]:
        """Lists all Unit Tests of the current user. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

            unit_test_list = client.modelci.list_unit_tests()

        Returns:
            A list of UnitTest objects.
        """
        response = self._connection.make_request(
            {},
            "modelci/unit_test",
            requests_command=requests.get,
        )
        return [
            UnitTest(test_id, self._connection)
            for test_id in response["unit_test_ids"]
        ]

    def delete_unit_test(self, unit_test_id: str) -> bool:
        """Deletes a Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            success = client.modelci.delete_unit_test(unit_test.id)

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

            import nucleus
            from nucleus.modelci.unit_test import ThresholdComparison
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            unit_test.add_metric(
                eval_function_id="ef_c61595wh49km7ppkk14g",
                threshold=0.5,
                threshold_comparison=ThresholdComparison.GREATER_THAN
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
