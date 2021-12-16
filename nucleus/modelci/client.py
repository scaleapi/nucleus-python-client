from typing import List

from nucleus.connection import Connection
from nucleus.job import AsyncJob

from .constants import UNIT_TEST_ID_KEY
from .data_transfer_objects.eval_function import (
    EvaluationCriterion,
    GetEvalFunctions,
)
from .data_transfer_objects.unit_test import CreateUnitTestRequest
from .errors import CreateUnitTestError
from .eval_functions.available_eval_functions import AvailableEvalFunctions
from .unit_test import UnitTest

SUCCESS_KEY = "success"
EVAL_FUNCTIONS_KEY = "eval_functions"


class ModelCI:
    """Model CI Python Client extension."""

    def __init__(self, api_key: str, endpoint: str):
        self.connection = Connection(api_key, endpoint)

    def __repr__(self):
        return f"ModelCI(connection='{self.connection}')"

    def __eq__(self, other):
        return self.connection == other.connection

    @property
    def eval_functions(self) -> AvailableEvalFunctions:
        """List all available evaluation functions which can be used to set up evaluation criteria.::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

            unit_test_criterion = client.modelci.eval_functions.bbox_iou() > 0.5  # Creates an EvaluationCriterion by comparison

        Returns:
            :class:`AvailableEvalFunctions`: A container for all the available eval functions
        """
        response = self.connection.get(
            "modelci/eval_fn",
        )
        payload = GetEvalFunctions.parse_obj(response)
        return AvailableEvalFunctions(payload.eval_functions)

    def create_unit_test(
        self,
        name: str,
        slice_id: str,
        evaluation_criteria: List[EvaluationCriterion],
    ) -> UnitTest:
        """Creates a new Unit Test from an existing Nucleus :class:`Slice`:. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

            unit_test = client.modelci.create_unit_test(
                name="sample_unit_test",
                slice_id="YOUR_SLICE_ID",
                evaluation_criteria=[client.modelci.eval_functions.bbox_iou() > 0.5]
            )

        Args:
            name: unique name of test
            slice_id: id of (pre-defined) slice of items to evaluate test on.
            evaluation_criteria: Pass/fail criteria for the test. Created with a comparison with an eval functions.
                See :class:`eval_functions`.

        Returns:
            Created UnitTest object.
        """
        if not evaluation_criteria:
            raise CreateUnitTestError(
                "Must pass an evaluation_criteria to the unit test! I.e. "
                "evaluation_criteria = [client.modelci.eval_functions.bbox_iou() > 0.5]"
            )
        response = self.connection.post(
            CreateUnitTestRequest(
                name=name,
                slice_id=slice_id,
                evaluation_criteria=evaluation_criteria,
            ).dict(),
            "modelci/unit_test",
        )
        return UnitTest(response[UNIT_TEST_ID_KEY], self.connection)

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
        response = self.connection.get(
            "modelci/unit_test",
        )
        return [
            UnitTest(test_id, self.connection)
            for test_id in response["unit_test_ids"]
        ]

    def delete_unit_test(self, unit_test_id: str) -> bool:
        """Deletes a Unit Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            unit_test = client.modelci.list_unit_tests()[0]

            success = client.modelci.create_unit_test(unit_test.id)

        Args:
            unit_test_id: unique ID of unit test

        Returns:
            Whether deletion was successful.
        """
        response = self.connection.delete(
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

            job = client.modelci.evaluate_model_on_unit_tests(
                model_id=model.id,
                unit_test_names=["sample_unit_test"],
            )
            job.sleep_until_complete() # Not required. Will block and update on status of the job.

        Args:
            model_id: ID of model to evaluate
            unit_test_names: list of unit test names of test to evaluate

        Returns:
            AsyncJob object of evaluation job
        """
        response = self.connection.post(
            {"test_names": unit_test_names},
            f"modelci/{model_id}/evaluate",
        )
        return AsyncJob.from_json(response, self.connection)
