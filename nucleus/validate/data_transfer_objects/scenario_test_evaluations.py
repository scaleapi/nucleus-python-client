from typing import Optional

from pydantic import root_validator, validator

from nucleus.pydantic_base import ImmutableModel


class EvaluationResult(ImmutableModel):
    item_ref_id: Optional[str] = None
    scene_ref_id: Optional[str] = None
    score: float = 0
    weight: float = 1

    @root_validator()
    def is_item_or_scene_provided(
        cls, values
    ):  # pylint: disable=no-self-argument
        if (
            values.get("item_ref_id") is None
            and values.get("scene_ref_id") is None
        ) or (
            (
                values.get("item_ref_id") is not None
                and values.get("scene_ref_id") is not None
            )
        ):
            raise ValueError("Must provide either item_ref_id or scene_ref_id")
        return values

    @validator("score", "weight")
    def is_normalized(cls, v):  # pylint: disable=no-self-argument
        if 0 <= v <= 1:
            return v
        raise ValueError(
            f"Expected evaluation score and weights to be normalized between 0 and 1, but got: {v}"
        )
