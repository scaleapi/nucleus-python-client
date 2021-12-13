from typing import List, Optional

from pydantic import validator

from ...pydantic_base import ImmutableModel
from ..constants import ThresholdComparison


class EvaluationCriterion(ImmutableModel):
    """
    An Evaluation Criterion is defined as an evaluation function, threshold, and comparator.
    It describes how to apply an evaluation function

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
    """ Expected format from GET modelci/eval_fn"""

    eval_functions: List[EvalFunctionEntry]
