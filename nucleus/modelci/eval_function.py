"""Data types for Evaluation Functions."""
from pydantic import BaseModel


class EvalFunctionResponse(BaseModel):
    """Encapsulates information about an evaluation function for Model CI."""

    id: str
    name: str
    user_id: str
    serialized_fn: str
    raw_source: str
    is_public: bool
