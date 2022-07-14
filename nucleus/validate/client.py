from typing import List

from nucleus.connection import Connection
from nucleus.job import AsyncJob

from .constants import EVAL_FUNCTION_KEY, SCENARIO_TEST_ID_KEY
from .data_transfer_objects.eval_function import (
    CreateEvalFunction,
    EvalFunctionEntry,
    GetEvalFunctions,
)
from .data_transfer_objects.scenario_test import (
    CreateScenarioTestRequest,
    EvalFunctionListEntry,
)
from .errors import CreateScenarioTestError
from .eval_functions.available_eval_functions import AvailableEvalFunctions
from .eval_functions.base_eval_function import EvalFunctionConfig
from .scenario_test import ScenarioTest

SUCCESS_KEY = "success"
EVAL_FUNCTIONS_KEY = "eval_functions"


class Validate:
    """Model CI Python Client extension."""

    def __init__(self, api_key: str, endpoint: str):
        self.connection = Connection(api_key, endpoint)

    def __repr__(self):
        return f"Validate(connection='{self.connection}')"

    def __eq__(self, other):
        return self.connection == other.connection

    @property
    def eval_functions(self) -> AvailableEvalFunctions:
        """List all available evaluation functions which can be used to set up evaluation criteria.::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

            # Creates an EvaluationCriterion by using a comparison op
            scenario_test_criterion = client.validate.eval_functions.bbox_iou() > 0.5

        Returns:
            :class:`AvailableEvalFunctions`: A container for all the available eval functions
        """
        response = self.connection.get(
            "validate/eval_fn",
        )
        payload = GetEvalFunctions.parse_obj(response)
        return AvailableEvalFunctions(payload.eval_functions)

    def create_scenario_test(
        self,
        name: str,
        slice_id: str,
        evaluation_functions: List[EvalFunctionConfig],
    ) -> ScenarioTest:
        """Creates a new Scenario Test from an existing Nucleus :class:`Slice`:. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")

            scenario_test = client.validate.create_scenario_test(
                name="sample_scenario_test",
                slice_id="YOUR_SLICE_ID",
                evaluation_functions=[client.validate.eval_functions.bbox_iou()]
            )

        Args:
            name: unique name of test
            slice_id: id of (pre-defined) slice of items to evaluate test on.
            evaluation_functions: :class:`EvalFunctionEntry` defines an evaluation metric for the test.
            Created with an element from the list of available eval functions. See :class:`eval_functions`.

        Returns:
            Created ScenarioTest object.
        """
        if not evaluation_functions:
            raise CreateScenarioTestError(
                "Must pass an evaluation_function to the scenario test! I.e. "
                "evaluation_functions=[client.validate.eval_functions.bbox_iou()]"
            )

        external_fns = [
            f.eval_func_entry.is_external_function
            for f in evaluation_functions
        ]
        if any(external_fns):
            assert all(
                external_fns
            ), "Cannot create scenario tests with mixed placeholder and non-placeholder evaluation functions"

        response = self.connection.post(
            CreateScenarioTestRequest(
                name=name,
                slice_id=slice_id,
                evaluation_functions=[
                    EvalFunctionListEntry(
                        id=ef.id, eval_func_arguments=ef.eval_func_arguments
                    )
                    for ef in evaluation_functions
                ],
            ).dict(),
            "validate/scenario_test",
        )
        return ScenarioTest(response[SCENARIO_TEST_ID_KEY], self.connection)

    def get_scenario_test(self, scenario_test_id: str) -> ScenarioTest:
        response = self.connection.get(
            f"validate/scenario_test/{scenario_test_id}",
        )
        return ScenarioTest(response["unit_test"]["id"], self.connection)

    @property
    def scenario_tests(self) -> List[ScenarioTest]:
        """Lists all Scenario Tests of the current user. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.create_scenario_test(
                "sample_scenario_test", "slc_bx86ea222a6g057x4380"
            )

            client.validate.scenario_tests

        Returns:
            A list of ScenarioTest objects.
        """
        response = self.connection.get(
            "validate/scenario_test",
        )
        return [
            ScenarioTest(test_id, self.connection)
            for test_id in response["scenario_test_ids"]
        ]

    def delete_scenario_test(self, scenario_test_id: str) -> bool:
        """Deletes a Scenario Test. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.scenario_tests[0]

            success = client.validate.delete_scenario_test(scenario_test.id)

        Args:
            scenario_test_id: unique ID of scenario test

        Returns:
            Whether deletion was successful.
        """
        response = self.connection.delete(
            f"validate/scenario_test/{scenario_test_id}",
        )
        return response[SUCCESS_KEY]

    def evaluate_model_on_scenario_tests(
        self, model_id: str, scenario_test_names: List[str]
    ) -> AsyncJob:
        """Evaluates the given model on the specified Scenario Tests. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            model = client.list_models()[0]
            scenario_test = client.validate.create_scenario_test(
                "sample_scenario_test", "slc_bx86ea222a6g057x4380"
            )

            job = client.validate.evaluate_model_on_scenario_tests(
                model_id=model.id,
                scenario_test_names=["sample_scenario_test"],
            )
            job.sleep_until_complete() # Not required. Will block and update on status of the job.

        Args:
            model_id: ID of model to evaluate
            scenario_test_names: list of scenario test names of test to evaluate

        Returns:
            AsyncJob object of evaluation job
        """
        response = self.connection.post(
            {"test_names": scenario_test_names},
            f"validate/{model_id}/evaluate",
        )
        return AsyncJob.from_json(response, self.connection)

    def create_external_eval_function(
        self,
        name: str,
    ) -> EvalFunctionEntry:
        """Creates a new external evaluation function. This external function can be used to upload evaluation
        results with functions defined and computed by the customer, without having to share the source code of the
        respective function.

        Args:
            name: unique name of evaluation function

        Raises:
            - NucleusAPIError if the creation of the function fails on the server side
            - ValidationError if the evaluation name is not well defined

        Returns:
            Created EvalFunctionConfig object.

        """

        response = self.connection.post(
            CreateEvalFunction(
                name=name,
                is_external_function=True,
                serialized_fn=None,
                raw_source=None,
            ).dict(),
            "validate/eval_fn",
        )
        return EvalFunctionEntry.parse_obj(response[EVAL_FUNCTION_KEY])
