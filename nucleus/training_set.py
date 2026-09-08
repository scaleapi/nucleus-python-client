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

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse

from nucleus.constants import (
    CREATED_AT_KEY,
    CREATED_BY_USER_ID_KEY,
    DATASET_COUNT_KEY,
    DATASET_ID_KEY,
    DATASET_ITEM_ID_KEY,
    DESCRIPTION_KEY,
    IMAGE_LOCATION_KEY,
    ITEM_COUNT_KEY,
    METADATA_KEY,
    NAME_KEY,
    PARENT_TRAINING_SET_ID_KEY,
    POINTCLOUD_LOCATION_KEY,
    REFERENCE_ID_KEY,
    STATUS_KEY,
    TRAINING_SET_ID_KEY,
    VERSION_LABEL_KEY,
    VERSION_MAJOR_KEY,
    VERSION_MINOR_KEY,
)
from nucleus.data_transfer_object.training_set import TrainingSetItemsPage

# Reuse the model-weights streamer: it retries transient failures and rejects a
# short read against Content-Length (so an interrupted transfer never gets
# promoted to a complete-looking file) — properties a local copy kept dropping.
from nucleus.model_weights import _stream_weights_to_file

if TYPE_CHECKING:
    from nucleus import NucleusClient
    from nucleus.dataset_item import DatasetItem


def _safe_filename_component(value: str) -> str:
    """Turn a member id into a single safe filename component.

    ``reference_id`` is user-supplied and unique only *within* a dataset, so it
    may contain path separators (``camera_a/frame`` vs ``camera_b/frame``) or
    traversal (``../x``). Collapse every separator to ``_`` (which keeps two
    distinct reference ids distinct instead of ``basename`` mapping them to the
    same name) and drop leading/trailing dots, yielding a component that cannot
    escape its parent directory.
    """
    flattened = str(value).replace("\\", "/").replace("/", "_")
    return flattened.strip().strip(".")


@dataclass
class TrainingSet:
    """A training set: a mutable, versioned, model-scoped set of dataset items."""

    id: str
    name: str
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

    def export_items(self, *, limit: int = 1000) -> List["DatasetItem"]:
        """Export every member as a fully-hydrated dataset item.

        Unlike :meth:`items` (which returns only the member ids one page at a
        time), this pages the whole training set and returns
        :class:`~nucleus.dataset_item.DatasetItem` objects carrying media
        locations, ``reference_id``, ``metadata``, ``width`` / ``height`` and
        the server-side ``dataset_item_id``.

        Parameters:
            limit: Page size for the underlying export requests.

        Returns:
            List[:class:`~nucleus.dataset_item.DatasetItem`]: Every member item.
        """
        if self._client is None:
            raise RuntimeError(
                "TrainingSet has no client; use NucleusClient.get_training_set."
            )
        return self._client.export_training_set_items(self.id, limit=limit)

    def export_to_file(self, path: str, *, limit: int = 1000) -> int:
        """Export every member to a JSONL file, returning the count written.

        Each line is a JSON object with the full export record —
        ``dataset_item_id``, ``dataset_id``, ``reference_id``, ``metadata``,
        ``image_location``, ``pointcloud_location``, ``width`` and ``height``.

        We deliberately write the raw export records rather than
        :func:`nucleus.utils.serialize_and_write` /
        ``DatasetItem.to_json``: that path drops ``dataset_item_id`` and
        ``dataset_id`` and asserts an ``image_location`` (so it would raise on
        pointcloud members). Writing the records ourselves keeps the export
        faithful to what the backend returned.

        Parameters:
            path: Destination JSONL path (parent dirs are created).
            limit: Page size for the underlying export requests.

        Returns:
            int: The number of records written.
        """
        if self._client is None:
            raise RuntimeError(
                "TrainingSet has no client; use NucleusClient.get_training_set."
            )
        directory = os.path.dirname(path)
        os.makedirs(directory or ".", exist_ok=True)
        count = 0
        # Stream records straight to disk so an arbitrarily large training set
        # is never fully resident in memory. The backend export record already
        # carries exactly the documented fields, so write it verbatim.
        with open(path, "w", encoding="utf-8") as file_pointer:
            for record in self._client.iter_training_set_export_records(
                self.id, limit=limit
            ):
                file_pointer.write(json.dumps(record) + "\n")
                count += 1
        return count

    def download_items(
        self, directory: str, *, limit: int = 1000, progress: bool = True
    ) -> int:
        """Download each member's media file into ``directory``.

        Streams each member's ``image_location`` (or ``pointcloud_location``
        for lidar members) to disk. Files are written under a per-``dataset_id``
        subdirectory and named by a sanitized ``reference_id`` (falling back to
        the globally-unique ``dataset_item_id``) plus the media URL's extension.
        Because ``reference_id`` is user-supplied and unique only *within* a
        dataset, any two members that would otherwise resolve to the same path
        (colliding ``reference_id``s within a dataset, or names that sanitize to
        the same string) are disambiguated with ``dataset_item_id`` so a
        download never silently overwrites an earlier member. Members with no
        media URL are skipped.

        A destination path that already exists on disk (e.g. from a previous
        call) is overwritten: downloading the same training set into the same
        directory is idempotent. Point at a fresh directory to avoid clobbering
        unrelated files.

        Records are paged lazily, so an arbitrarily large training set is never
        fully resident in memory.

        Parameters:
            directory: Destination directory (created if missing).
            limit: Page size for the underlying export requests.
            progress: Show a tqdm progress bar while downloading.

        Returns:
            int: The number of media files downloaded.
        """
        if self._client is None:
            raise RuntimeError(
                "TrainingSet has no client; use NucleusClient.get_training_set."
            )
        os.makedirs(directory, exist_ok=True)

        records: Any = self._client.iter_training_set_export_records(
            self.id, limit=limit
        )
        if progress:
            from tqdm import tqdm

            records = tqdm(records, desc="Downloading training set items")

        count = 0
        used_paths: set = set()
        for record in records:
            url = record.get(IMAGE_LOCATION_KEY) or record.get(
                POINTCLOUD_LOCATION_KEY
            )
            if not url:
                continue
            dataset_item_id = record.get(DATASET_ITEM_ID_KEY)
            reference_id = record.get(REFERENCE_ID_KEY)
            name = (
                _safe_filename_component(reference_id) if reference_id else ""
            ) or (
                _safe_filename_component(dataset_item_id)
                if dataset_item_id
                else ""
            )
            if not name:
                continue
            extension = os.path.splitext(urlparse(url).path)[1]
            dataset_id = record.get(DATASET_ID_KEY)
            subdir = (
                os.path.join(directory, _safe_filename_component(dataset_id))
                if dataset_id
                else directory
            )
            path = os.path.join(subdir, f"{name}{extension}")
            # Guarantee no member silently overwrites another. On a collision,
            # disambiguate with the globally-unique dataset_item_id, then a
            # numeric suffix as a last resort (in case that fallback path is
            # itself already taken), until the path is unused.
            if path in used_paths:
                stem = (
                    _safe_filename_component(dataset_item_id)
                    if dataset_item_id
                    else name
                )
                path = os.path.join(subdir, f"{stem}{extension}")
                suffix = 1
                while path in used_paths:
                    path = os.path.join(subdir, f"{stem}_{suffix}{extension}")
                    suffix += 1
            used_paths.add(path)
            os.makedirs(subdir, exist_ok=True)
            _stream_weights_to_file(url, path)
            count += 1
        return count

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
            wait_for_completion=wait_for_completion,
            verbose=verbose,
        )
        return self.refresh()

    def remove_items(
        self,
        item_ids: Optional[List[str]] = None,
        *,
        items: Optional[List[Dict[str, str]]] = None,
        slice_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        slice_ids: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        training_set_ids: Optional[List[str]] = None,
        wait_for_completion: bool = True,
        verbose: bool = True,
    ) -> "TrainingSet":
        """Remove members from this training set.

        Accepts the same sources as :meth:`add_items`: explicit ``item_ids`` /
        ``items``, or every member covered by a whole slice / dataset / training
        set. Unknown ids are ignored. See
        :meth:`NucleusClient.remove_training_set_items` for details.

        Returns:
            self, refreshed.
        """
        if self._client is None:
            raise RuntimeError("TrainingSet has no client.")
        self._client.remove_training_set_items(
            self.id,
            item_ids=item_ids,
            items=items,
            slice_id=slice_id,
            dataset_id=dataset_id,
            slice_ids=slice_ids,
            dataset_ids=dataset_ids,
            training_set_ids=training_set_ids,
            wait_for_completion=wait_for_completion,
            verbose=verbose,
        )
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
