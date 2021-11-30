from pydantic import BaseModel, validator

from ..constants import ThresholdComparison


class EvalFunctionCondition(BaseModel):
    # TODO: Having only eval_function_id hurts readability -> Add function name
    eval_function_id: str
    threshold_comparison: ThresholdComparison
    threshold: float

    @validator("eval_function_id")
    def valid_eval_function_id(cls, v):
        if not v.startswith("ef_"):
            raise ValueError(f"Expected field to start with 'ef_', got '{v}'")
        return v
