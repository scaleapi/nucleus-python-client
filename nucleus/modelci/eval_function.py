"""Data types for Evaluation Functions."""
from dataclasses import dataclass


@dataclass
class EvalFunction:
    """Encapsulates an evaluation function for Model CI."""

    id: str
    name: str
    user_id: str
    serialized_fn: str
    raw_source: str
    is_public: bool
