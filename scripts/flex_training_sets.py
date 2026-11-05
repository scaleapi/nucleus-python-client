#!/usr/bin/env python
"""One big guarded flex of the full training-set surface (DE-8692).

Walks *every* training-set method the SDK added, against a local Nucleus
backend, in one pass. Every step is guarded (a failure is recorded and the run
keeps going) so all kinks surface together rather than one-at-a-time.

Coverage:
  create_model                         -> throwaway destination model
  create_training_set(item_ids)        -> root v1.0
  get_training_set / refresh
  list_training_set_items / .items     -> incl. limit/offset pagination
  get_model_training_set               -> the model's pinned set
  update (name/description/metadata)
  add_training_set_items (item_ids)
  add_training_set_items (items= pairs) -> (dataset_id, reference_id) form
  remove_training_set_items (item_ids)
  create_training_set_version / .new_version  -> child (prune + add)
  create_training_set(parent=..., bump_type="major")  -> v2.0 sibling
  create_training_set(training_set_ids=[...])  -> unversioned union ("mix")
  family()                             -> lineage listing
  repin_training_set + get_model_training_set  -> verify the pin moved
  export_training_set_items / export_items
  export_to_file (JSONL)
  download_items (media to disk)
  list_training_sets                   -> (known-suspect: unscoped list route)
  input validation guards (expected ValueErrors)
  delete_training_set                  -> cleanup

Usage:
    # uses $NUCLEUS_PRODCLONE_API_KEY and the local prodclone backend by default
    python scripts/flex_training_sets.py

    # explicit key / endpoint / model
    python scripts/flex_training_sets.py --api-key <key> \
        --endpoint http://localhost:3000/v1/nucleus --model prj_...
"""
import argparse
import os
import tempfile
import time
import traceback
from typing import List

from nucleus import NucleusClient
from nucleus.training_set import TrainingSet

# The six dataset items from the request. They live on ds_d6ng124awzzg0gtc8x50.
ITEM_IDS = [
    "di_d6ng124z6a1006gm86ag",
    "di_d6ng124z6a1006gm86b0",
    "di_d6ng124z6a1006gm86bg",
    "di_d6ng124z6a1006gm86c0",
    "di_d6ng124z6a1006gm86e0",
    "di_d6ng124z6a1006gm86eg",
]


class Flex:
    """Runs each stage guarded, collecting issues for a final report."""

    def __init__(self) -> None:
        self.issues: List[str] = []

    def step(self, label: str, fn):
        print(f"\n=== {label} ===")
        try:
            result = fn()
            print(f"[ok] {label}")
            return result
        except Exception as exc:  # noqa: BLE001 -- surface everything
            self.issues.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None

    def expect_raises(self, label: str, exc_type, fn):
        """Invert a step: the call SHOULD raise exc_type. Records an issue if
        it doesn't, or raises the wrong type."""
        print(f"\n=== {label} (expect {exc_type.__name__}) ===")
        try:
            fn()
        except exc_type as exc:
            print(f"[ok] {label}: raised {type(exc).__name__} as expected")
            return
        except Exception as exc:  # noqa: BLE001
            self.issues.append(
                f"{label}: raised {type(exc).__name__} (wanted {exc_type.__name__}): {exc}"
            )
            print(
                f"[FAIL] {label}: wrong exception {type(exc).__name__}: {exc}"
            )
            return
        self.issues.append(f"{label}: did NOT raise {exc_type.__name__}")
        print(f"[FAIL] {label}: no exception raised")

    def report(self) -> int:
        print("\n" + "=" * 60)
        if not self.issues:
            print("No issues -- the training-set flex completed cleanly.")
            return 0
        print(f"{len(self.issues)} issue(s) encountered:")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")
        return 1


def members(ts: TrainingSet) -> List[str]:
    """Sorted member ids of a training set (paged to 1000)."""
    return sorted(ts.items(limit=1000).item_ids)


def show(label: str, ts: TrainingSet) -> None:
    print(
        f"  {label}: {ts.id} v{ts.version_major}.{ts.version_minor} "
        f"status={ts.status} count={ts.item_count}"
    )


def main() -> None:  # noqa: C901 -- deliberately linear, one call per stage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NUCLEUS_PRODCLONE_API_KEY"),
        help="Scale API key (default: $NUCLEUS_PRODCLONE_API_KEY).",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:3000/v1/nucleus",
        help="Nucleus API base URL (default: local prodclone backend).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Existing model id (prj_*) to scope the sets to. Default: create "
        "a fresh throwaway model.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Skip the final delete_training_set cleanup.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error(
            "no API key: pass --api-key or set NUCLEUS_PRODCLONE_API_KEY"
        )

    client = NucleusClient(api_key=args.api_key, endpoint=args.endpoint)
    flex = Flex()

    # ------------------------------------------------------------------ #
    # 0. Destination model (fresh throwaway unless --model given).
    # ------------------------------------------------------------------ #
    if args.model:
        model = client.get_model(args.model)
        print(f"Using existing model {model.id} ({model.name!r})")
    else:
        ref = f"ts-flex-{int(time.time())}"
        model = client.create_model(
            name="training-set flex",
            reference_id=ref,
            metadata={"purpose": "DE-8692 training-set end-to-end flex"},
        )
        print(f"Created model {model.id} (reference_id={ref!r})")

    # ------------------------------------------------------------------ #
    # 1. create_training_set(item_ids) -> root v1.0. Via Model wrapper.
    # ------------------------------------------------------------------ #
    root = flex.step(
        "1. model.create_training_set(item_ids=[4]) -> root",
        lambda: model.create_training_set(
            "ts-flex-root",
            item_ids=ITEM_IDS[:4],
            description="root set",
            metadata={"stage": "root"},
            wait_for_completion=True,
        ),
    )
    if root is None:
        raise SystemExit(flex.report())
    show("root", root)
    print(f"  members: {members(root)}")

    # ------------------------------------------------------------------ #
    # 2. get_training_set + refresh round-trip.
    # ------------------------------------------------------------------ #
    flex.step("2a. get_training_set", lambda: client.get_training_set(root.id))
    flex.step("2b. TrainingSet.refresh", lambda: root.refresh())

    # ------------------------------------------------------------------ #
    # 3. items() pagination: limit/offset must page cleanly.
    # ------------------------------------------------------------------ #
    def check_pagination():
        full = root.items(limit=1000)
        assert full.total == 4, f"total={full.total}, want 4"
        p1 = root.items(limit=2, offset=0)
        p2 = root.items(limit=2, offset=2)
        got = set(p1.item_ids) | set(p2.item_ids)
        assert len(p1.item_ids) == 2, f"page1 size {len(p1.item_ids)}"
        assert got == set(
            full.item_ids
        ), f"paged {got} != full {set(full.item_ids)}"
        return f"total={full.total}, two pages of 2 reunion OK"

    flex.step("3. items() limit/offset pagination", check_pagination)

    # ------------------------------------------------------------------ #
    # 4. get_model_training_set -> model's pinned set is the root.
    # ------------------------------------------------------------------ #
    flex.step(
        "4. get_model_training_set (pinned == root?)",
        lambda: (
            lambda pinned: (
                show("pinned", pinned),
                None
                if pinned.id == root.id
                else (_ for _ in ()).throw(
                    AssertionError(f"pinned {pinned.id} != root {root.id}")
                ),
            )
        )(client.get_model_training_set(model)),
    )

    # ------------------------------------------------------------------ #
    # 5. update name/description/metadata.
    # ------------------------------------------------------------------ #
    def do_update():
        updated = root.update(
            name="ts-flex-root-renamed",
            description="updated description",
            metadata={"stage": "root", "updated": True},
        )
        assert updated.name == "ts-flex-root-renamed", updated.name
        assert (
            updated.description == "updated description"
        ), updated.description
        assert updated.metadata.get("updated") is True, updated.metadata
        return updated.name

    flex.step("5. update(name/description/metadata)", do_update)

    # ------------------------------------------------------------------ #
    # 6. add_items (explicit item_ids) -> 4 -> 5.
    # ------------------------------------------------------------------ #
    def do_add():
        root.add_items(item_ids=[ITEM_IDS[4]])  # add e0
        m = members(root)
        assert ITEM_IDS[4] in m, f"{ITEM_IDS[4]} not added: {m}"
        assert len(m) == 5, f"want 5, got {len(m)}: {m}"
        return f"count now {len(m)}"

    flex.step("6. add_items(item_ids=[e0]) 4->5", do_add)

    # ------------------------------------------------------------------ #
    # 7. add_items via (dataset_id, reference_id) pair form.
    #    Discover a pair for the last uncovered item (eg) from the export.
    # ------------------------------------------------------------------ #
    def do_add_pair():
        # eg (ITEM_IDS[5]) isn't a member yet. Resolve its (dataset_id,
        # reference_id) pair from an existing member's dataset (they share one),
        # then add eg by pair to exercise the items= form.
        sample = client.export_training_set_records(root.id)[0]
        dataset_id = sample["dataset_id"]
        eg_item = client.get_dataset(dataset_id).loc(ITEM_IDS[5])["item"]
        eg_ref = eg_item.reference_id
        if not eg_ref:
            raise RuntimeError("could not resolve reference_id for eg")
        root.add_items(
            items=[{"dataset_id": dataset_id, "reference_id": eg_ref}]
        )
        m = members(root)
        assert ITEM_IDS[5] in m, f"eg not added via pair: {m}"
        return f"added eg via (dataset_id={dataset_id}, reference_id={eg_ref}); count {len(m)}"

    flex.step("7. add_items(items=[{dataset_id,reference_id}])", do_add_pair)

    # ------------------------------------------------------------------ #
    # 8. remove_items (explicit item_ids).
    # ------------------------------------------------------------------ #
    def do_remove():
        before = members(root)
        root.remove_items(item_ids=[ITEM_IDS[3]])  # remove c0
        after = members(root)
        assert ITEM_IDS[3] not in after, f"c0 still present: {after}"
        assert len(after) == len(before) - 1, f"{before} -> {after}"
        return f"count {len(before)} -> {len(after)}"

    flex.step("8. remove_items(item_ids=[c0])", do_remove)

    # ------------------------------------------------------------------ #
    # 9. new_version(prune + add) -> child v1.1.
    # ------------------------------------------------------------------ #
    parent_members = set(members(root))
    child = flex.step(
        "9. new_version(prune ag, add c0) -> child",
        lambda: root.new_version(
            item_ids=[ITEM_IDS[3]],  # re-add c0
            removed_item_ids=[ITEM_IDS[0]],  # prune ag
            version_label="child-rc1",
            wait_for_completion=True,
        ),
    )
    if child is not None:
        show("child", child)
        cm = set(members(child))
        expected = (parent_members | {ITEM_IDS[3]}) - {ITEM_IDS[0]}
        print(f"  child members: {sorted(cm)}")
        if cm != expected:
            flex.issues.append(
                f"9. new_version membership: got {sorted(cm)} want {sorted(expected)}"
            )
            print(
                f"  [FAIL] membership {sorted(cm)} != expected {sorted(expected)}"
            )

    # ------------------------------------------------------------------ #
    # 10. create_training_set(parent=..., bump_type="major") -> v2.0.
    # ------------------------------------------------------------------ #
    major = flex.step(
        "10. create_training_set(parent=root, bump_type=major) -> v2.0",
        lambda: client.create_training_set(
            "ts-flex-major",
            model=model,
            parent_training_set_id=root.id,
            bump_type="major",
            wait_for_completion=True,
        ),
    )
    if major is not None:
        show("major", major)

    # ------------------------------------------------------------------ #
    # 11. create_training_set(training_set_ids=[...]) -> unversioned union.
    # ------------------------------------------------------------------ #
    mix = None
    if child is not None:
        mix = flex.step(
            "11. create_training_set(training_set_ids=[root, child]) -> mix",
            lambda: client.create_training_set(
                "ts-flex-mix",
                model=model,
                training_set_ids=[root.id, child.id],
                wait_for_completion=True,
            ),
        )
        if mix is not None:
            show("mix", mix)
            mm = set(members(mix))
            union = set(members(root)) | set(members(child))
            print(f"  mix members: {sorted(mm)}")
            if mm != union:
                flex.issues.append(
                    f"11. mix union: got {sorted(mm)} want {sorted(union)}"
                )
                print(f"  [FAIL] union {sorted(mm)} != {sorted(union)}")

    # ------------------------------------------------------------------ #
    # 12. family() -> lineage listing (root + child + major share a root).
    # ------------------------------------------------------------------ #
    def do_family():
        fam = root.family()
        ids = {t.id for t in fam}
        print(
            f"  family members: {[(t.id, f'v{t.version_major}.{t.version_minor}') for t in fam]}"
        )
        must = {root.id}
        if child is not None:
            must.add(child.id)
        if major is not None:
            must.add(major.id)
        missing = must - ids
        assert not missing, f"family missing {missing}"
        return f"{len(fam)} versions in lineage"

    flex.step("12. family() lineage", do_family)

    # ------------------------------------------------------------------ #
    # 13. repin_training_set -> move the model's pin to the child, verify.
    # ------------------------------------------------------------------ #
    if child is not None:

        def do_repin():
            model.repin_training_set(child.id)
            pinned = client.get_model_training_set(model)
            assert (
                pinned.id == child.id
            ), f"pinned {pinned.id} != child {child.id}"
            return f"model now pinned to {pinned.id}"

        flex.step("13. repin_training_set(child) + verify", do_repin)

    # ------------------------------------------------------------------ #
    # 14. export_items (hydrated DatasetItems).
    # ------------------------------------------------------------------ #
    def do_export_items():
        items = root.export_items()
        assert (
            len(items) == root.refresh().item_count
        ), f"export {len(items)} != item_count {root.item_count}"
        sample = items[0]
        assert sample.reference_id, "export item missing reference_id"
        return (
            f"{len(items)} hydrated items; sample ref={sample.reference_id!r}"
        )

    flex.step("14. export_items()", do_export_items)

    # ------------------------------------------------------------------ #
    # 15. export_to_file (JSONL).
    # ------------------------------------------------------------------ #
    tmpdir = tempfile.mkdtemp(prefix="ts-flex-")

    def do_export_file():
        path = os.path.join(tmpdir, "root.jsonl")
        n = root.export_to_file(path)
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        assert len(lines) == n, f"wrote {n} but file has {len(lines)} lines"
        return f"{n} records -> {path}"

    flex.step("15. export_to_file(JSONL)", do_export_file)

    # ------------------------------------------------------------------ #
    # 16. download_items (media to disk).
    # ------------------------------------------------------------------ #
    def do_download():
        dest = os.path.join(tmpdir, "media")
        try:
            n = root.download_items(dest, progress=False)
        except RuntimeError as exc:
            # The prodclone export returns raw (unsigned) scale-us-attachments S3
            # URLs that aren't fetchable from a dev box, so the stream 403s. That
            # is an environment/data limitation, not an SDK defect (the method
            # faithfully streams whatever URL the backend returned) — skip rather
            # than fail so this env-only gap doesn't mask real regressions.
            if "403" in str(exc):
                print(
                    f"  [skip] download_items: storage 403 (unsigned local S3 URL) — {exc}"
                )
                return "skipped (environmental S3 403)"
            raise
        files = []
        for dirpath, _, fnames in os.walk(dest):
            files += [os.path.join(dirpath, f) for f in fnames]
        assert n == len(files), f"reported {n} but {len(files)} files on disk"
        sizes = [os.path.getsize(f) for f in files]
        assert all(s > 0 for s in sizes), f"empty file(s): {sizes}"
        return f"{n} media files, sizes {sizes}"

    flex.step("16. download_items(media)", do_download)

    # ------------------------------------------------------------------ #
    # 17. list_training_sets (unscoped list route -- suspect 401).
    # ------------------------------------------------------------------ #
    flex.step(
        "17. list_training_sets (unscoped list route)",
        lambda: f"{len(client.list_training_sets())} sets visible",
    )

    # ------------------------------------------------------------------ #
    # 18. Input-validation guards (should raise ValueError before any HTTP).
    # ------------------------------------------------------------------ #
    flex.expect_raises(
        "18a. create_training_set no source",
        ValueError,
        lambda: client.create_training_set("no-source", model=model),
    )
    flex.expect_raises(
        "18b. removed_item_ids without parent",
        ValueError,
        lambda: client.create_training_set(
            "bad",
            model=model,
            item_ids=ITEM_IDS[:1],
            removed_item_ids=[ITEM_IDS[0]],
        ),
    )
    flex.expect_raises(
        "18c. add_training_set_items no source",
        ValueError,
        lambda: client.add_training_set_items(root.id),
    )

    # ------------------------------------------------------------------ #
    # 19. delete_training_set (cleanup the mix set).
    # ------------------------------------------------------------------ #
    if mix is not None and not args.keep:

        def do_delete():
            client.delete_training_set(mix.id)
            # A follow-up GET should now fail.
            try:
                client.get_training_set(mix.id)
            except Exception:  # noqa: BLE001
                return f"{mix.id} deleted (GET now fails as expected)"
            raise AssertionError(f"{mix.id} still fetchable after delete")

        flex.step("19. delete_training_set(mix) + verify gone", do_delete)

    print("\nDone.")
    print(f"  model: {model.id}")
    print(f"  root:  {root.id}")
    if child is not None:
        print(f"  child: {child.id}")
    if major is not None:
        print(f"  major: {major.id}")
    print(f"  tmp artifacts: {tmpdir}")
    raise SystemExit(flex.report())


if __name__ == "__main__":
    main()
