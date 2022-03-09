from typing import List

from nucleus.pydantic_base import ImmutableModel


class EvalDetail(ImmutableModel):
    id: str


class GetEvalHistory(ImmutableModel):
    evaluations: List[EvalDetail]
