"""Evaluation V2 presets — saved, reusable evaluation configurations.

A preset bundles a ``name`` with a label configuration (``rollup_groups``)
and ``exclusion_rules`` so the same configuration can be applied across many
evaluations. Presets are private to the creating user.

Create and manage presets via :class:`~nucleus.NucleusClient`::

    preset = client.create_evaluation_v2_preset(
        "vehicles",
        rollup_groups=[RollupGroup("vehicle", ["car", "truck"])],
        exclusion_rules=[LabelExclusionRule(scope="item", target="prediction", labels=["ignore"])],
    )
    client.create_benchmark_evaluation_v2(benchmark_id, model_run_id, preset=preset)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from nucleus.constants import (
    CREATED_AT_KEY,
    CREATED_BY_USER_ID_KEY,
    DELETED_AT_KEY,
    EXCLUSION_RULES_CAMEL_KEY,
    EXCLUSION_RULES_KEY,
    ID_KEY,
    NAME_KEY,
    ROLLUP_GROUPS_CAMEL_KEY,
    ROLLUP_GROUPS_KEY,
    UPDATED_AT_KEY,
)
from nucleus.evaluation_v2 import (
    RollupGroup,
    _parse_json_field,
    _parse_rollup_groups,
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
    rollup_groups: Optional[List[RollupGroup]] = None
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
            id=str(payload[ID_KEY]),
            name=str(payload[NAME_KEY]),
            rollup_groups=_parse_rollup_groups(
                _parse_json_field(
                    payload.get(ROLLUP_GROUPS_KEY)
                    if payload.get(ROLLUP_GROUPS_KEY) is not None
                    else payload.get(ROLLUP_GROUPS_CAMEL_KEY)
                )
            ),
            exclusion_rules=_parse_json_field(
                payload.get(EXCLUSION_RULES_KEY)
                if payload.get(EXCLUSION_RULES_KEY) is not None
                else payload.get(EXCLUSION_RULES_CAMEL_KEY)
            ),
            created_by_user_id=payload.get(CREATED_BY_USER_ID_KEY),
            created_at=payload.get(CREATED_AT_KEY),
            updated_at=payload.get(UPDATED_AT_KEY),
            deleted_at=payload.get(DELETED_AT_KEY),
            _client=client,
        )

    def update(
        self,
        *,
        name: Any = _UNSET,
        rollup_groups: Any = _UNSET,
        exclusion_rules: Any = _UNSET,
    ) -> "EvaluationV2Preset":
        """Update this preset in place.

        Only the arguments you pass are changed. Passing
        ``rollup_groups=None`` / ``exclusion_rules=None`` clears that field;
        omitting an argument leaves it unchanged.

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
            rollup_groups=rollup_groups,
            exclusion_rules=exclusion_rules,
        )
        self.__dict__.update(updated.__dict__)
        return self

    def delete(self) -> None:
        """Delete this preset."""
        if self._client is None:
            raise RuntimeError("EvaluationV2Preset has no client.")
        self._client.delete_evaluation_v2_preset(self.id)
