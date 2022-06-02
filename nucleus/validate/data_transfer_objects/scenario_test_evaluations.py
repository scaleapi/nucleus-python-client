from typing import List

from pydantic import validator

from nucleus.pydantic_base import ImmutableModel


class EvalDetail(ImmutableModel):
    id: str


class GetEvalHistory(ImmutableModel):
    evaluations: List[EvalDetail]


class EvaluationResult(ImmutableModel):
    item_ref_id: str
    score: float
    weight: float = 1

    @validator("score", "weight")
    def is_normalized(cls, v):  # pylint: disable=no-self-argument
        if 0 <= v <= 1:
            return v
        raise ValueError(
            f"Expected evaluation score and weights to be normalized between 0 and 1, but got: {v}"
        )
