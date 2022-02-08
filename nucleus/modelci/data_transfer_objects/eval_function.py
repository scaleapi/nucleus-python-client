from typing import List, Optional

from pydantic import validator

from ...pydantic_base import ImmutableModel
from ..constants import ThresholdComparison


class EvaluationCriterion(ImmutableModel):
    """
    An Evaluation Criterion is defined as an evaluation function, threshold, and comparator.
    It describes how to apply an evaluation function

    Notes:
        To define the evaluation criteria for a scenario test we've created some syntactic sugar to make it look closer to an
        actual function call, and we also hide away implementation details related to our data model that simply are not clear,
        UX-wise.

        Instead of defining criteria like this::

            from nucleus.modelci.data_transfer_objects.eval_function import (
                EvaluationCriterion,
                ThresholdComparison,
            )

            criteria = [
                EvaluationCriterion(
                    eval_function_id="ef_c6m1khygqk400918ays0",  # bbox_recall
                    threshold_comparison=ThresholdComparison.GREATER_THAN,
                    threshold=0.5,
                ),
            ]

        we define it like this::

            bbox_recall = client.modelci.eval_functions.bbox_recall
            criteria = [
                bbox_recall() > 0.5
            ]

        The chosen method allows us to document the available evaluation functions in an IDE friendly fashion and hides away
        details like internal IDs (`"ef_...."`).

        The actual `EvaluationCriterion` is created by overloading the comparison operators for the base class of an evaluation
        function. Instead of the comparison returning a bool, we've made it create an `EvaluationCriterion` with the correct
        signature to send over the wire to our API.


    Parameters:
        eval_function_id (str): ID of evaluation function
        threshold_comparison (:class:`ThresholdComparison`): comparator for evaluation. i.e. threshold=0.5 and threshold_comparator > implies that a test only passes if score > 0.5.
        threshold (float): numerical threshold that together with threshold comparison, defines success criteria for test evaluation.
    """

    # TODO: Having only eval_function_id hurts readability -> Add function name
    eval_function_id: str
    threshold_comparison: ThresholdComparison
    threshold: float

    @validator("eval_function_id")
    def valid_eval_function_id(cls, v):  # pylint: disable=no-self-argument
        if not v.startswith("ef_"):
            raise ValueError(f"Expected field to start with 'ef_', got '{v}'")
        return v


class EvalFunctionEntry(ImmutableModel):
    """Encapsulates information about an evaluation function for Model CI."""

    id: str
    name: str
    is_public: bool
    user_id: str
    serialized_fn: str
    raw_source: Optional[str] = None


class GetEvalFunctions(ImmutableModel):
    """Expected format from GET modelci/eval_fn"""

    eval_functions: List[EvalFunctionEntry]
