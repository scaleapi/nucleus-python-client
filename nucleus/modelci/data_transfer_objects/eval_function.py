from typing import List, Optional

from pydantic import BaseModel, validator

from ..constants import ThresholdComparison


class EvalFunctionCondition(BaseModel):
    """
    eval_function_id: ID of evaluation function
    threshold_comparison: comparator for evaluation. i.e. threshold=0.5 and threshold_comparator > implies that a test only passes if score > 0.5.
    threshold: numerical threshold that together with threshold comparison, defines success criteria for test evaluation.
    """

    # TODO: Having only eval_function_id hurts readability -> Add function name
    eval_function_id: str
    threshold_comparison: ThresholdComparison
    threshold: float

    @validator("eval_function_id")
    def valid_eval_function_id(cls, v):
        if not v.startswith("ef_"):
            raise ValueError(f"Expected field to start with 'ef_', got '{v}'")
        return v


class EvalFunctionEntry(BaseModel):
    """Encapsulates information about an evaluation function for Model CI."""

    id: str
    name: str
    user_id: str
    serialized_fn: str
    is_public: bool
    raw_source: Optional[str] = None


class GetEvalFunctions(BaseModel):
    eval_functions: List[EvalFunctionEntry]
