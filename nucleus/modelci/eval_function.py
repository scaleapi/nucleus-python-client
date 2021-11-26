"""Data types for Evaluation Functions."""
from typing import Optional

from pydantic import BaseModel


class EvalFunctionDefinition(BaseModel):
    """Encapsulates information about an evaluation function for Model CI."""

    id: str
    name: str
    user_id: str
    serialized_fn: str
    is_public: bool
    raw_source: Optional[str] = None
