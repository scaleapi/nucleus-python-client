"""Evaluation V2 presets — saved, reusable evaluation configurations.

A preset bundles a ``name`` with ``allowed_label_matches`` and ``exclusion_rules``
so the same configuration can be applied across many evaluations. Presets are
private to the creating user.

Mirrors the ``/v1/nucleus/evaluationV2Presets`` REST endpoints on the backend.
Create and manage presets via :class:`~nucleus.NucleusClient`::

    preset = client.create_evaluation_v2_preset(
        "vehicles",
        allowed_label_matches=[AllowedLabelMatch("car", "vehicle")],
        exclusion_rules=[LabelExclusionRule(scope="item", target="prediction", labels=["ignore"])],
    )
    client.create_evaluation_v2(model_run_id, preset=preset)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from nucleus.evaluation_v2 import (
    AllowedLabelMatch,
    _parse_json_field,
    parse_allowed_label_matches,
)

if TYPE_CHECKING:
    from nucleus import NucleusClient


# Sentinel distinguishing "argument omitted" from an explicit ``None`` (which,
# for ``exclusion_rules`` on update, means "clear the rules").
class _Unset:
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<UNSET>"


_UNSET = _Unset()


@dataclass
class EvaluationV2Preset:
    """A saved Evaluation V2 configuration owned by the current user."""

    id: str
    name: str
    allowed_label_matches: Optional[List[AllowedLabelMatch]] = None
    exclusion_rules: Optional[List[Dict[str, Any]]] = None
    created_by_user_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None
    _client: Optional["NucleusClient"] = field(repr=False, default=None)

    @classmethod
    def from_json(
        cls,
        payload: Dict[str, Any],
        client: Optional["NucleusClient"] = None,
    ) -> "EvaluationV2Preset":
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            allowed_label_matches=parse_allowed_label_matches(
                payload.get("allowed_label_matches")
            ),
            exclusion_rules=_parse_json_field(payload.get("exclusion_rules")),
            created_by_user_id=payload.get("created_by_user_id"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            deleted_at=payload.get("deleted_at"),
            _client=client,
        )

    def update(
        self,
        *,
        name: Any = _UNSET,
        allowed_label_matches: Any = _UNSET,
        exclusion_rules: Any = _UNSET,
    ) -> "EvaluationV2Preset":
        """Update this preset in place.

        Only the arguments you pass are changed. Passing
        ``exclusion_rules=None`` clears the rules; omitting it leaves them
        unchanged.

        Returns:
            self, with updated fields.
        """
        if self._client is None:
            raise RuntimeError(
                "EvaluationV2Preset has no client; fetch it via "
                "NucleusClient.list_evaluation_v2_presets."
            )
        updated = self._client.update_evaluation_v2_preset(
            self.id,
            name=name,
            allowed_label_matches=allowed_label_matches,
            exclusion_rules=exclusion_rules,
        )
        self.__dict__.update(updated.__dict__)
        return self

    def delete(self) -> None:
        """Delete this preset."""
        if self._client is None:
            raise RuntimeError("EvaluationV2Preset has no client.")
        self._client.delete_evaluation_v2_preset(self.id)
