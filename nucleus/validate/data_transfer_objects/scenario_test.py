from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    # Backwards compatibility is even uglier with mypy
    from pydantic.v1 import validator
else:
    try:
        # NOTE: we always use pydantic v1 but have to do these shenanigans to support both v1 and v2
        from pydantic.v1 import validator
    except ImportError:
        from pydantic import validator

from nucleus.pydantic_base import ImmutableModel


class EvalFunctionListEntry(ImmutableModel):
    id: str
    eval_func_arguments: dict


class CreateScenarioTestRequest(ImmutableModel):
    name: str
    slice_id: str
    evaluation_functions: List[EvalFunctionListEntry]

    @validator("slice_id")  # pylint: disable=used-before-assignment
    def startswith_slice_indicator(cls, v):  # pylint: disable=no-self-argument
        if not v.startswith("slc_"):
            raise ValueError(f"Expected field to start with 'slc_', got '{v}'")
        return v
