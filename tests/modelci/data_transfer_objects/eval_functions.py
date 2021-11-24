from typing import List

from pydantic import BaseModel

from nucleus.modelci import EvalFunctionResponse


class GetEvalFunctions(BaseModel):
    eval_functions: List[EvalFunctionResponse]
