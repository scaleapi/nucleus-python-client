"""Training sets — mutable, versioned, model-scoped dataset-item collections.

A training set is a named collection of ``dataset_item`` ids (spanning one or
more source datasets) scoped to a single model. Unlike a benchmark it is
**mutable**: items can be added and removed after creation, and each edit /
re-cut can be captured as a new **version** so a model's training data is
reproducible over time.

Create and manage training sets via :class:`~nucleus.NucleusClient` or a
:class:`~nucleus.model.Model`::

    training_set = model.create_training_set(
        "pedestrians-v1", slice_id="slc_..."
    )
    training_set.add_items(dataset_ids=["ds_..."])
    v2 = training_set.new_version(removed_item_ids=["di_bad"], bump_type="major")
    model.repin_training_set(v2.id)  # point the model at the new version

A training set spans datasets: give it members through explicit ``item_ids``,
``(dataset_id, reference_id)`` pairs, whole slices/datasets, or by unioning in
the members of **other training sets** via ``training_set_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from nucleus.constants import (
    CREATED_AT_KEY,
    CREATED_BY_USER_ID_KEY,
    DATASET_COUNT_KEY,
    DESCRIPTION_KEY,
    ITEM_COUNT_KEY,
    METADATA_KEY,
    MODEL_ID_KEY,
    NAME_KEY,
    PARENT_TRAINING_SET_ID_KEY,
    STATUS_KEY,
    TRAINING_SET_ID_KEY,
    VERSION_LABEL_KEY,
    VERSION_MAJOR_KEY,
    VERSION_MINOR_KEY,
)
from nucleus.data_transfer_object.training_set import TrainingSetItemsPage

if TYPE_CHECKING:
    from nucleus import NucleusClient


@dataclass
class TrainingSet:
    """A training set: a mutable, versioned, model-scoped set of dataset items."""

    id: str
    name: str
    #: The model this training set is scoped to.
    model_id: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_by_user_id: Optional[str] = None
    created_at: Optional[str] = None
    item_count: Optional[int] = None
    dataset_count: Optional[int] = None
    #: Lifecycle status: ``"building"`` (create/add job still streaming members
    #: in), ``"ready"``, or ``"failed"``.
    status: Optional[str] = None
    #: Lineage: this training set's parent version (``None`` for a root set).
    parent_training_set_id: Optional[str] = None
    #: Version of this training set relative to its lineage root (root is 1.0).
    version_major: Optional[int] = None
    version_minor: Optional[int] = None
    #: Optional human-readable version label (e.g. ``"rc1"``, ``"holdout-v2"``).
    version_label: Optional[str] = None
    _client: Optional["NucleusClient"] = field(repr=False, default=None)

    @classmethod
    def from_json(
        cls,
        payload: Dict[str, Any],
        client: Optional["NucleusClient"] = None,
    ) -> "TrainingSet":
        return cls(
            id=str(payload[TRAINING_SET_ID_KEY]),
            name=str(payload[NAME_KEY]),
            model_id=payload.get(MODEL_ID_KEY),
            description=payload.get(DESCRIPTION_KEY),
            metadata=payload.get(METADATA_KEY),
            created_by_user_id=payload.get(CREATED_BY_USER_ID_KEY),
            created_at=payload.get(CREATED_AT_KEY),
            item_count=payload.get(ITEM_COUNT_KEY),
            dataset_count=payload.get(DATASET_COUNT_KEY),
            status=payload.get(STATUS_KEY),
            parent_training_set_id=payload.get(PARENT_TRAINING_SET_ID_KEY),
            version_major=payload.get(VERSION_MAJOR_KEY),
            version_minor=payload.get(VERSION_MINOR_KEY),
            version_label=payload.get(VERSION_LABEL_KEY),
            _client=client,
        )

    def refresh(self) -> "TrainingSet":
        """Reload this training set from Nucleus.

        Returns:
            self, with updated fields.
        """
        if self._client is None:
            raise RuntimeError(
                "TrainingSet has no client; use NucleusClient.get_training_set."
            )
        updated = self._client.get_training_set(self.id)
        self.__dict__.update(updated.__dict__)
        return self

    def update(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TrainingSet":
        """Update this training set's name, description, or metadata.

        Only the arguments you pass are changed. To change membership, use
        :meth:`add_items` / :meth:`remove_items` (or cut a :meth:`new_version`).

        Returns:
            self, with updated fields.
        """
        if self._client is None:
            raise RuntimeError("TrainingSet has no client.")
        updated = self._client.update_training_set(
            self.id,
            name=name,
            description=description,
            metadata=metadata,
        )
        self.__dict__.update(updated.__dict__)
        return self

    def delete(self) -> None:
        """Delete this training set."""
        if self._client is None:
            raise RuntimeError("TrainingSet has no client.")
        self._client.delete_training_set(self.id)

    def items(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> TrainingSetItemsPage:
        """Return one page of this training set's member item ids.

        Parameters:
            limit: Optional page size.
            offset: Optional offset for pagination.

        Returns:
            :class:`~nucleus.data_transfer_object.training_set.TrainingSetItemsPage`:
            The page of dataset item ids and the total member count.
        """
        if self._client is None:
            raise RuntimeError("TrainingSet has no client.")
        return self._client.list_training_set_items(
            self.id, limit=limit, offset=offset
        )

    def add_items(
        self,
        *,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        training_set_ids: Optional[List[str]] = None,
        scene_ids: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> "TrainingSet":
        """Add items to this training set.

        See :meth:`NucleusClient.add_training_set_items` for parameter details.

        Returns:
            self, refreshed.
        """
        if self._client is None:
            raise RuntimeError("TrainingSet has no client.")
        self._client.add_training_set_items(
            self.id,
            item_ids=item_ids,
            items=items,
            slice_id=slice_id,
            dataset_id=dataset_id,
            slice_ids=slice_ids,
            dataset_ids=dataset_ids,
            training_set_ids=training_set_ids,
            scene_ids=scene_ids,
            wait_for_completion=wait_for_completion,
            verbose=verbose,
        )
        return self.refresh()

    def remove_items(self, item_ids: List[str]) -> "TrainingSet":
        """Remove items from this training set.

        Unknown ids are ignored.

        Parameters:
            item_ids: Dataset item ids (``di_*``) to remove.

        Returns:
            self, refreshed.
        """
        if self._client is None:
            raise RuntimeError("TrainingSet has no client.")
        self._client.remove_training_set_items(self.id, item_ids)
        return self.refresh()

    def new_version(
        self,
        *,
        item_ids: Optional[List[str]] = None,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        training_set_ids: Optional[List[str]] = None,
        removed_item_ids: Optional[List[str]] = None,
        bump_type: Optional[str] = None,
        version_major: Optional[int] = None,
        version_minor: Optional[int] = None,
        version_label: Optional[str] = None,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> "TrainingSet":
        """Create a new **version** downstream of this training set.

        The child inherits this set's items, the source arguments add on top,
        and ``removed_item_ids`` prune inherited items
        (``final set = parent ∪ added ∖ removed``). See
        :meth:`NucleusClient.create_training_set_version` for details.

        Returns:
            :class:`TrainingSet`: The newly created version (a distinct object).
        """
        if self._client is None:
            raise RuntimeError("TrainingSet has no client.")
        return self._client.create_training_set_version(
            self.id,
            item_ids=item_ids,
            items=items,
            slice_id=slice_id,
            dataset_id=dataset_id,
            slice_ids=slice_ids,
            dataset_ids=dataset_ids,
            training_set_ids=training_set_ids,
            removed_item_ids=removed_item_ids,
            bump_type=bump_type,
            version_major=version_major,
            version_minor=version_minor,
            version_label=version_label,
            wait_for_completion=wait_for_completion,
            verbose=verbose,
        )

    def family(self) -> List["TrainingSet"]:
        """Return every version in this training set's lineage (its family).

        Returns:
            List of :class:`TrainingSet` sharing this set's lineage root.
        """
        if self._client is None:
            raise RuntimeError("TrainingSet has no client.")
        return self._client.list_training_set_family(self.id)
