from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Backwards compatibility is even uglier with mypy
    from pydantic.v1 import root_validator, validator
else:
    try:
        # NOTE: we always use pydantic v1 but have to do these shenanigans to support both v1 and v2
        from pydantic.v1 import root_validator, validator
    except ImportError:
        from pydantic import root_validator, validator

from nucleus.pydantic_base import ImmutableModel


class EvaluationResult(ImmutableModel):
    track_ref_id: Optional[str] = None
    item_ref_id: Optional[str] = None
    scene_ref_id: Optional[str] = None
    score: float = 0
    weight: float = 1

    @root_validator()  # pylint: disable=used-before-assignment
    def is_item_or_scene_provided(
        cls, values
    ):  # pylint: disable=no-self-argument
        ref_ids = [
            values.get("track_ref_id", None),
            values.get("item_ref_id", None),
            values.get("scene_ref_id", None),
        ]
        if len([ref_id for ref_id in ref_ids if ref_id is not None]) != 1:
            raise ValueError(
                "Must provide exactly one of track_ref_id, item_ref_id, or scene_ref_id"
            )
        return values

    @validator("weight")  # pylint: disable=used-before-assignment
    def is_normalized(cls, v):  # pylint: disable=no-self-argument
        if 0 <= v <= 1:
            return v
        raise ValueError(
            f"Expected evaluation weights to be normalized between 0 and 1, but got: {v}"
        )
