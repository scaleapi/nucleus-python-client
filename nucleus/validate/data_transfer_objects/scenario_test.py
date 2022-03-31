from typing import List

from pydantic import validator

from nucleus.pydantic_base import ImmutableModel


class EvalFunctionListEntry(ImmutableModel):
    id: str
    eval_func_arguments: dict


class CreateScenarioTestRequest(ImmutableModel):
    name: str
    slice_id: str
    evaluation_functions: List[EvalFunctionListEntry]

    @validator("slice_id")
    def startswith_slice_indicator(cls, v):  # pylint: disable=no-self-argument
        if not v.startswith("slc_"):
            raise ValueError(f"Expected field to start with 'slc_', got '{v}'")
        return v
