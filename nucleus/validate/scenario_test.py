"""Scenario Tests combine collections of data and evaluation metrics to accelerate model evaluation.

With Model CI Scenario Tests, an ML engineer can define a Scenario Test from critical
edge case scenarios that the model must get right (e.g. pedestrians at night),
and have confidence that they’re always shipping the best model.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ..connection import Connection
from ..constants import (
    DATASET_ITEMS_KEY,
    NAME_KEY,
    SCENES_KEY,
    SLICE_ID_KEY,
    TRACKS_KEY,
)
from ..dataset_item import DatasetItem
from ..scene import Scene
from ..track import Track
from .constants import (
    EVAL_FUNCTION_ID_KEY,
    SCENARIO_TEST_ID_KEY,
    SCENARIO_TEST_METRICS_KEY,
    THRESHOLD_COMPARISON_KEY,
    THRESHOLD_KEY,
    EntityLevel,
    ThresholdComparison,
)
from .data_transfer_objects.scenario_test_evaluations import EvaluationResult
from .data_transfer_objects.scenario_test_metric import AddScenarioTestFunction
from .eval_functions.available_eval_functions import (
    EvalFunction,
    ExternalEvalFunction,
)
from .scenario_test_evaluation import ScenarioTestEvaluation
from .scenario_test_metric import ScenarioTestMetric


@dataclass
class ScenarioTest:
    """A Scenario Test combines a slice and at least one evaluation criterion. A :class:`ScenarioTest` is not created through
    the default constructor but using the instructions shown in :class:`Validate`. This :class:`ScenarioTest` class only
    simplifies the interaction with the scenario tests from this SDK.

    Attributes:
        id (str): The ID of the scenario test.
        connection (Connection): The connection to Nucleus API.
        name (str): The name of the scenario test.
        slice_id (str): The ID of the associated Nucleus slice.
    """

    id: str
    connection: Connection = field(repr=False)
    name: str = field(init=False)
    slice_id: str = field(init=False)
    baseline_model_id: Optional[str] = None

    @classmethod
    def from_id(cls, unit_test_id: str, connection: Connection):
        # TODO(gunnar): Remove this pattern. It's too slow. We should get all the info required in one call
        response = connection.get(
            f"validate/scenario_test/{unit_test_id}/info",
        )
        instance = cls(unit_test_id, connection)
        instance.name = response[NAME_KEY]
        instance.slice_id = response[SLICE_ID_KEY]
        return instance

    @classmethod
    def from_response(cls, response, connection: Connection):
        instance = cls(response["id"], connection)
        instance.name = response[NAME_KEY]
        instance.slice_id = response[SLICE_ID_KEY]
        instance.baseline_model_id = response.get("baseline_model_id", None)
        return instance

    def add_eval_function(
        self, eval_function: EvalFunction
    ) -> ScenarioTestMetric:
        """Creates and adds a new evaluation metric to the :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.create_scenario_test(
                "sample_scenario_test", "slc_bx86ea222a6g057x4380"
            )

            e = client.validate.eval_functions
            # Assuming a user would like to add all available public evaluation functions as criteria
            scenario_test.add_eval_function(
                e.bbox_iou
            )
            scenario_test.add_eval_function(
                e.bbox_map
            )
            scenario_test.add_eval_function(
                e.bbox_precision
            )
            scenario_test.add_eval_function(
                e.bbox_recall
            )

        Args:
            eval_function: :class:`EvalFunction`

        Raises:
            NucleusAPIError: By adding this function, the scenario test mixes external with non-external functions which is not permitted.

        Returns:
            The created ScenarioTestMetric object.
        """

        response = self.connection.post(
            AddScenarioTestFunction(
                scenario_test_name=self.name,
                eval_function_id=eval_function.id,
            ).dict(),
            "validate/scenario_test_eval_function",
        )

        return ScenarioTestMetric(
            scenario_test_id=response[SCENARIO_TEST_ID_KEY],
            eval_function_id=response[EVAL_FUNCTION_ID_KEY],
            threshold=response.get(THRESHOLD_KEY, None),
            threshold_comparison=response.get(
                THRESHOLD_COMPARISON_KEY,
                ThresholdComparison.GREATER_THAN_EQUAL_TO,
            ),
            connection=self.connection,
        )

    def get_eval_functions(self) -> List[ScenarioTestMetric]:
        """Retrieves all criteria of the :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.scenario_tests[0]

            scenario_test.get_eval_functions()

        Returns:
            A list of ScenarioTestMetric objects.
        """
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/metrics",
        )
        return [
            ScenarioTestMetric(**metric, connection=self.connection)
            for metric in response[SCENARIO_TEST_METRICS_KEY]
        ]

    def get_eval_history(self) -> List[ScenarioTestEvaluation]:
        """Retrieves evaluation history for :class:`ScenarioTest`. ::

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.scenario_tests[0]

            scenario_test.get_eval_history()

        Returns:
            A list of :class:`ScenarioTestEvaluation` objects.
        """
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/eval_history/details",
        )
        evaluations = [
            ScenarioTestEvaluation.from_request(eval_payload, self.connection)
            for eval_payload in response
        ]
        return evaluations

    def get_items(
        self, level: EntityLevel = EntityLevel.ITEM
    ) -> Union[List[Track], List[DatasetItem], List[Scene]]:
        """Gets items within a scenario test at a given level, returning a list of Track, DatasetItem, or Scene objects.

        Args:
            level: :class:`EntityLevel`

        Returns:
            A list of :class:`ScenarioTestEvaluation` objects.
        """
        response = self.connection.get(
            f"validate/scenario_test/{self.id}/items",
        )
        if level == EntityLevel.TRACK:
            return [
                Track.from_json(track, connection=self.connection)
                for track in response.get(TRACKS_KEY, [])
            ]
        if level == EntityLevel.SCENE:
            return [
                Scene.from_json(scene, skip_validate=True)
                for scene in response.get(SCENES_KEY, [])
            ]
        if level == EntityLevel.ITEM:
            return [
                DatasetItem.from_json(item)
                for item in response.get(DATASET_ITEMS_KEY, [])
            ]
        raise ValueError(f"Invalid entity level: {level}")

    def set_baseline_model(self, model_id: str):
        """Sets a new baseline model for the ScenarioTest.  In order to be eligible to be a baseline,
        this scenario test must have been evaluated using that model.  The baseline model's performance
        is used as the threshold for all metrics against which other models are compared.

            import nucleus
            client = nucleus.NucleusClient("YOUR_SCALE_API_KEY")
            scenario_test = client.validate.scenario_tests[0]

            scenario_test.set_baseline_model('my_baseline_model_id')

        Returns:
            A list of :class:`ScenarioTestEvaluation` objects.
        """
        response = self.connection.post(
            {},
            f"validate/scenario_test/{self.id}/set_baseline_model/{model_id}",
        )
        self.baseline_model_id = response.get("baseline_model_id")
        return self.baseline_model_id

    def upload_external_evaluation_results(
        self,
        eval_fn: ExternalEvalFunction,
        results: List[EvaluationResult],
        model_id: str,
    ):
        assert (
            eval_fn.eval_func_entry.is_external_function
        ), "Submitting evaluation results is only available for external functions."

        assert (
            len(results) > 0
        ), "Submitting evaluation requires at least one result."

        level: Optional[EntityLevel] = None
        metric_per_ref_id = {}
        weight_per_ref_id = {}
        aggregate_weighted_sum = 0.0
        aggregate_weight = 0.0

        # Ensures reults at only one EntityLevel are provided, otherwise throwing a ValueError
        def ensure_level_consistency_or_raise(
            cur_level: Optional[EntityLevel], new_level: EntityLevel
        ):
            if level is not None and level != new_level:
                raise ValueError(
                    f"All evaluation results must only pertain to one level. Received {cur_level} then {new_level}"
                )

        # aggregation based on https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
        for r in results:
            # Ensure results are uploaded ONLY for ONE OF tracks, items, and scenes
            if r.track_ref_id is not None:
                ensure_level_consistency_or_raise(level, EntityLevel.TRACK)
                level = EntityLevel.TRACK
            if r.item_ref_id is not None:
                ensure_level_consistency_or_raise(level, EntityLevel.ITEM)
                level = EntityLevel.ITEM
            if r.scene_ref_id is not None:
                ensure_level_consistency_or_raise(level, EntityLevel.SCENE)
                level = EntityLevel.SCENE
            ref_id = (
                r.track_ref_id
                if level == EntityLevel.TRACK
                else (
                    r.item_ref_id
                    if level == EntityLevel.ITEM
                    else r.scene_ref_id
                )
            )

            # Aggregate scores and weights
            metric_per_ref_id[ref_id] = r.score
            weight_per_ref_id[ref_id] = r.weight
            aggregate_weighted_sum += r.score * r.weight
            aggregate_weight += r.weight

        payload = {
            "unit_test_id": self.id,
            "eval_function_id": eval_fn.id,
            "result_per_ref_id": metric_per_ref_id,
            "weight_per_ref_id": weight_per_ref_id,
            "overall_metric": aggregate_weighted_sum / aggregate_weight,
            "model_id": model_id,
            "slice_id": self.slice_id,
            "level": level.value if level else None,
        }
        response = self.connection.post(
            payload,
            "validate/scenario_test/upload_results",
        )
        return response
