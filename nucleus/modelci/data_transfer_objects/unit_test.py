from typing import List

from pydantic import BaseModel, validator

from .eval_function import EvaluationCriteria


class CreateUnitTestRequest(BaseModel):
    name: str
    slice_id: str
    evaluation_criteria: List[EvaluationCriteria]

    @validator("slice_id")
    def startswith_slice_indicator(cls, v):  # pylint: disable=no-self-argument
        if not v.startswith("slc_"):
            raise ValueError(f"Expected field to start with 'slc_', got '{v}'")
        return v
