from typing import List

from pydantic import BaseModel


class EvalDetail(BaseModel):
    id: str


class GetEvalHistory(BaseModel):
    evaluations: List[EvalDetail]
