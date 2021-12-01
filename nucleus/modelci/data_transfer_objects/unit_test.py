from typing import List

from pydantic import BaseModel, validator

from .eval_function import EvalFunctionCondition


class CreateUnitTestRequest(BaseModel):
    name: str
    slice_id: str
    evaluation_conditions: List[EvalFunctionCondition]

    @validator("slice_id")
    def startswith_slice_indicator(
        cls, v
    ):  # Pydantic says the first param should be `cls` not `self`
        if not v.startswith("slc_"):
            raise ValueError(f"Expected field to start with 'slc_', got '{v}'")
        return v
