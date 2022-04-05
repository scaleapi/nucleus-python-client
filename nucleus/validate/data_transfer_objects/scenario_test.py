from typing import List

from pydantic import validator

from nucleus.pydantic_base import ImmutableModel

from .eval_function import EvalFunctionEntry


class CreateScenarioTestRequest(ImmutableModel):
    name: str
    slice_id: str
    evaluation_functions: List[EvalFunctionEntry]

    @validator("slice_id")
    def startswith_slice_indicator(cls, v):  # pylint: disable=no-self-argument
        if not v.startswith("slc_"):
            raise ValueError(f"Expected field to start with 'slc_', got '{v}'")
        return v
