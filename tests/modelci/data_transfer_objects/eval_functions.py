from typing import List

from pydantic import BaseModel

from nucleus.modelci.eval_function import EvalFunctionDefinition


class GetEvalFunctions(BaseModel):
    eval_functions: List[EvalFunctionDefinition]
