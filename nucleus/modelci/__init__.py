"""Model CI Python Library."""
from typing import List


from nucleus.job import AsyncJob
from nucleus.connection import Connection

from .constants import (
    EVAL_FUNCTION_ID_KEY,
    ID_KEY,
    THRESHOLD_COMPARISON_KEY,
    THRESHOLD_KEY,
    UNIT_TEST_ID_KEY,
    UNIT_TEST_NAME_KEY,
    ThresholdComparison,
)
from .data_transfer_objects.unit_test import CreateUnitTestRequest
from .unit_test import (
    UnitTest,
    UnitTestInfo,
)
from .unit_test_metric import UnitTestMetric
from .unit_test_evaluation import (
    UnitTestEvaluation,
    UnitTestItemEvaluation,
)
from .utils import format_unit_test_eval_response
from nucleus.modelci.eval_functions.available_eval_functions import (
    AvailableEvalFunctions,
)
from .data_transfer_objects.eval_function import (
    EvalFunctionCondition,
    GetEvalFunctions,
    EvalFunctionEntry,
)

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

    @property
    def eval_functions(self) -> AvailableEvalFunctions:
        """List all available evaluation functions. ::

        import nucleus
        client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

        eval_functions = client.modelci.eval_functions

        Returns:
            :class:`AvailableEvalFunctions`: A container for all the available eval functions
        """
        response = self._connection.get(
            "modelci/eval_fn",
        )
        payload = GetEvalFunctions.parse_obj(response)
        return AvailableEvalFunctions(payload.eval_functions)

    def list_eval_functions(self) -> List[EvalFunctionEntry]:
        """List all available evaluation functions. ::

        import nucleus
        client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

        eval_functions = client.modelci.list_eval_functions()
        """
        response = self._connection.get(
            "modelci/eval_fn",
        )
        return [
            EvalFunctionEntry(**eval_function)
            for eval_function in response[EVAL_FUNCTIONS_KEY]
        ]

    def create_unit_test(
        self,
        name: str,
        slice_id: str,
        evalulation_conditions: List[EvalFunctionCondition],
    ) -> UnitTest:
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
        response = self._connection.post(
            CreateUnitTestRequest(
                name=name,
                slice_id=slice_id,
                evaluation_conditions=evalulation_conditions,
            ).dict(),
            "modelci/unit_test",
        )
        return UnitTest(response[UNIT_TEST_ID_KEY], self)

    def list_unit_tests(self) -> List[UnitTest]:
        """Lists all Unit Tests of the current user. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.create_unit_test(
                "sample_unit_test", "slc_bx86ea222a6g057x4380"
            )

            client.modelci.list_unit_tests()

        Returns:
            A list of UnitTest objects.
        """
        response = self._connection.get(
            "modelci/unit_test",
        )
        return [
            UnitTest(test_id, self) for test_id in response["unit_test_ids"]
        ]

    def delete_unit_test(self, unit_test_id: str) -> bool:
        """Creates a new Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            success = client.modelci.create_unit_test(unit_test.id)

        Args:
            unit_test_id: unique ID of unit test

        Returns:
            Whether deletion was successful.
        """
        response = self._connection.delete(
            f"modelci/unit_test/{unit_test_id}",
        )
        return response[SUCCESS_KEY]

    def evaluate_model_on_unit_tests(
        self, model_id: str, unit_test_names: List[str]
    ) -> AsyncJob:
        """Evaluates the given model on the specified Unit Tests. ::

            import nucleus
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
        response = self._connection.post(
            {"test_names": unit_test_names},
            f"modelci/{model_id}/evaluate",
        )
        return AsyncJob.from_json(response, self._connection)
