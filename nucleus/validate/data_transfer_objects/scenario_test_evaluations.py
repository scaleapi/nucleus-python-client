from typing import List, Optional

from pydantic import validator

from nucleus.pydantic_base import ImmutableModel


class EvalDetail(ImmutableModel):
    id: str


class GetEvalHistory(ImmutableModel):
    evaluations: List[EvalDetail]


class EvaluationResult(ImmutableModel):
    item_ref_id: str
    score: float
    weight: Optional[float]

    @validator("score")
    def normalized_score(cls, v):  # pylint: disable=no-self-argument
        if 0 <= v <= 1:
            return v
        raise ValueError(
            f"Expected evaluation score to be normalized between 0 and 1, but got: {v}"
        )
