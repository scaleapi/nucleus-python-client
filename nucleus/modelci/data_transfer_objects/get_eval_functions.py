from typing import List

from pydantic import BaseModel

from ..eval_function import EvalFunctionEntry


class GetEvalFunctions(BaseModel):
    eval_functions: List[EvalFunctionEntry]
