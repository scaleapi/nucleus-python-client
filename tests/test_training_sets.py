"""Unit tests for training sets (no live API)."""

from unittest.mock import MagicMock

import pytest
import requests

from nucleus import NucleusClient, TrainingSet

_TRAINING_SET_ROW = {
    "training_set_id": "ts_1",
    "name": "pedestrians",
    "model_id": "prj_1",
    "description": "desc",
    "metadata": {"team": "av"},
    "created_by_user_id": "u_1",
    "created_at": "2026-08-26T00:00:00.000Z",
    "item_count": 10,
    "dataset_count": 2,
    "status": "ready",
}


# --------------------------------------------------------------------------- #
# from_json
# --------------------------------------------------------------------------- #
def test_training_set_from_json_maps_training_set_id():
    ts = TrainingSet.from_json(_TRAINING_SET_ROW)
    assert ts.id == "ts_1"
    assert ts.name == "pedestrians"
    assert ts.model_id == "prj_1"
    assert ts.item_count == 10
    assert ts.dataset_count == 2
    assert ts.status == "ready"


def test_training_set_from_json_parses_lineage_fields():
    ts = TrainingSet.from_json(
        {
            **_TRAINING_SET_ROW,
            "parent_training_set_id": "ts_parent",
            "version_major": 2,
            "version_minor": 1,
            "version_label": "rc1",
        }
    )
    assert ts.parent_training_set_id == "ts_parent"
    assert ts.version_major == 2
    assert ts.version_minor == 1
    assert ts.version_label == "rc1"
    # Root training set: lineage fields absent.
    root = TrainingSet.from_json(_TRAINING_SET_ROW)
    assert root.parent_training_set_id is None
    assert root.version_major is None


# --------------------------------------------------------------------------- #
# Create (model-scoped, async)
# --------------------------------------------------------------------------- #
def _mock_async_create(client, *, row=None):
    """Wire up the async create flow: 202 {training_set_id, job_id} ->
    poll the build job -> re-fetch the ready training set."""
    client.connection.post = MagicMock(
        return_value={"training_set_id": "ts_1", "job_id": "job_1"}
    )
    client.get_job = (
        MagicMock()
    )  # .sleep_until_complete() is a no-op MagicMock
    client.get_training_set = MagicMock(
        return_value=TrainingSet.from_json(row or _TRAINING_SET_ROW, client)
    )
    return client


def test_create_training_set_from_slice_polls_then_returns_ready():
    client = _mock_async_create(NucleusClient(api_key="test"))
    training_set = client.create_training_set(
        "pedestrians", model="prj_1", description="desc", slice_id="slc_1"
    )
    payload, route = client.connection.post.call_args[0]
    assert route == "models/prj_1/trainingSet"
    assert payload == {
        "name": "pedestrians",
        "description": "desc",
        "slice_id": "slc_1",
    }
    client.get_job.assert_called_once_with("job_1")
    client.get_job.return_value.sleep_until_complete.assert_called_once()
    client.get_training_set.assert_called_once_with("ts_1")
    assert training_set.id == "ts_1"
    assert training_set.status == "ready"


def test_create_training_set_from_item_ids_and_metadata():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_training_set(
        "pedestrians",
        model="prj_1",
        metadata={"team": "av"},
        item_ids=["di_1", "di_2"],
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["item_ids"] == ["di_1", "di_2"]
    assert payload["metadata"] == {"team": "av"}


def test_create_training_set_from_items_pairs():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_training_set(
        "pedestrians",
        model="prj_1",
        items=[{"dataset_id": "ds_1", "reference_id": "ref_1"}],
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["items"] == [
        {"dataset_id": "ds_1", "reference_id": "ref_1"}
    ]


def test_create_training_set_from_training_set_ids_source():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_training_set(
        "merged", model="prj_1", training_set_ids=["ts_a", "ts_b"]
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["training_set_ids"] == ["ts_a", "ts_b"]


def test_create_training_set_combines_multiple_sources():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_training_set(
        "multi",
        model="prj_1",
        item_ids=["di_3"],
        slice_ids=["slc_1", "slc_2"],
        dataset_ids=["ds_1", "ds_2"],
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["item_ids"] == ["di_3"]
    assert payload["slice_ids"] == ["slc_1", "slc_2"]
    assert payload["dataset_ids"] == ["ds_1", "ds_2"]


def test_create_training_set_accepts_model_object():
    client = _mock_async_create(NucleusClient(api_key="test"))
    model = MagicMock()
    model.id = "prj_99"
    # isinstance(model, Model) is False for a MagicMock, so pass a real model id
    # via the .id attribute path by using an actual Model.
    from nucleus.model import Model

    real_model = Model("prj_99", "m", "ref", {}, client)
    client.create_training_set("m", model=real_model, item_ids=["di_1"])
    _, route = client.connection.post.call_args[0]
    assert route == "models/prj_99/trainingSet"


def test_create_training_set_requires_a_source():
    client = NucleusClient(api_key="test")
    with pytest.raises(ValueError, match="at least one"):
        client.create_training_set("ts", model="prj_1")


def test_create_training_set_no_wait_skips_polling():
    client = _mock_async_create(
        NucleusClient(api_key="test"),
        row={**_TRAINING_SET_ROW, "status": "building", "item_count": 0},
    )
    ts = client.create_training_set(
        "pedestrians",
        model="prj_1",
        slice_id="slc_1",
        wait_for_completion=False,
    )
    client.get_job.assert_not_called()
    client.get_training_set.assert_called_once_with("ts_1")
    assert ts.status == "building"


# --------------------------------------------------------------------------- #
# Versioning / lineage
# --------------------------------------------------------------------------- #
def test_create_training_set_version_payload_from_parent_kwarg():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_training_set(
        "pedestrians-v2",
        model="prj_1",
        parent_training_set_id="ts_parent",
        bump_type="major",
        removed_item_ids=["di_9"],
        item_ids=["di_1"],
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["parent_training_set_id"] == "ts_parent"
    assert payload["bump_type"] == "major"
    assert payload["removed_item_ids"] == ["di_9"]
    assert payload["item_ids"] == ["di_1"]


def test_create_training_set_parent_alone_is_a_valid_source():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_training_set(
        "reversion", model="prj_1", parent_training_set_id="ts_parent"
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["parent_training_set_id"] == "ts_parent"


def test_create_training_set_removed_items_requires_parent():
    client = NucleusClient(api_key="test")
    with pytest.raises(ValueError, match="removed_item_ids"):
        client.create_training_set(
            "ts", model="prj_1", item_ids=["di_1"], removed_item_ids=["di_9"]
        )


def test_create_training_set_version_endpoint_polls_and_refetches():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(
        return_value={"training_set_id": "ts_2", "job_id": "job_v"}
    )
    client.get_job = MagicMock()
    client.get_training_set = MagicMock(
        return_value=TrainingSet.from_json(
            {**_TRAINING_SET_ROW, "training_set_id": "ts_2"}, client
        )
    )
    new_version = client.create_training_set_version(
        "ts_1", removed_item_ids=["di_9"], bump_type="major"
    )
    payload, route = client.connection.post.call_args[0]
    assert route == "trainingSets/ts_1/versions"
    assert payload["removed_item_ids"] == ["di_9"]
    assert payload["bump_type"] == "major"
    client.get_job.assert_called_once_with("job_v")
    client.get_training_set.assert_called_once_with("ts_2")
    assert new_version.id == "ts_2"


def test_list_training_set_family():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value=[
            dict(_TRAINING_SET_ROW),
            {**_TRAINING_SET_ROW, "training_set_id": "ts_2"},
        ]
    )
    family = client.list_training_set_family("ts_1")
    client.connection.get.assert_called_once_with("trainingSets/ts_1/family")
    assert [ts.id for ts in family] == ["ts_1", "ts_2"]


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #
def test_list_training_sets():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value=[dict(_TRAINING_SET_ROW)])
    training_sets = client.list_training_sets()
    client.connection.get.assert_called_once_with("trainingSets")
    assert len(training_sets) == 1
    assert training_sets[0].id == "ts_1"


def test_get_training_set():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value=dict(_TRAINING_SET_ROW))
    ts = client.get_training_set("ts_1")
    client.connection.get.assert_called_once_with("trainingSets/ts_1")
    assert ts.id == "ts_1"


def test_get_model_training_set_reads_pinned():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value=dict(_TRAINING_SET_ROW))
    ts = client.get_model_training_set("prj_1")
    client.connection.get.assert_called_once_with("models/prj_1/trainingSet")
    assert ts.id == "ts_1"


def test_update_training_set_sends_only_provided_fields():
    client = NucleusClient(api_key="test")
    client.connection.patch = MagicMock(
        return_value={**_TRAINING_SET_ROW, "name": "renamed"}
    )
    ts = client.update_training_set("ts_1", name="renamed")
    payload, route = client.connection.patch.call_args[0]
    assert route == "trainingSets/ts_1"
    assert payload == {"name": "renamed"}
    assert ts.name == "renamed"


def test_delete_training_set():
    client = NucleusClient(api_key="test")
    client.connection.make_request = MagicMock(return_value=MagicMock())
    client.delete_training_set("ts_1")
    args = client.connection.make_request.call_args[0]
    assert args[1] == "trainingSets/ts_1"
    assert args[2] is requests.delete


# --------------------------------------------------------------------------- #
# Repin
# --------------------------------------------------------------------------- #
def test_repin_training_set_puts_model_route():
    client = NucleusClient(api_key="test")
    client.connection.put = MagicMock(return_value=dict(_TRAINING_SET_ROW))
    ts = client.repin_training_set("prj_1", "ts_1")
    payload, route = client.connection.put.call_args[0]
    assert route == "models/prj_1/trainingSet"
    assert payload == {"training_set_id": "ts_1"}
    assert ts.id == "ts_1"


# --------------------------------------------------------------------------- #
# Items: add / remove / list
# --------------------------------------------------------------------------- #
def test_add_training_set_items_posts_sources_and_polls():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(return_value={"job_id": "job_add"})
    client.get_job = MagicMock()
    client.add_training_set_items(
        "ts_1", item_ids=["di_1"], training_set_ids=["ts_9"]
    )
    payload, route = client.connection.post.call_args[0]
    assert route == "trainingSets/ts_1/items"
    assert payload["item_ids"] == ["di_1"]
    assert payload["training_set_ids"] == ["ts_9"]
    client.get_job.assert_called_once_with("job_add")


def test_add_training_set_items_requires_a_source():
    client = NucleusClient(api_key="test")
    with pytest.raises(ValueError, match="at least one"):
        client.add_training_set_items("ts_1")


def test_remove_training_set_items_deletes_with_body():
    client = NucleusClient(api_key="test")
    # A small explicit removal is synchronous: the response carries no job_id.
    client.connection.make_request = MagicMock(
        return_value={"training_set_id": "ts_1", "status": "ready", "job_id": None}
    )
    client.get_job = MagicMock()
    client.remove_training_set_items("ts_1", ["di_1", "di_2"])
    args = client.connection.make_request.call_args[0]
    assert args[0] == {"item_ids": ["di_1", "di_2"]}
    assert args[1] == "trainingSets/ts_1/items"
    assert args[2] is requests.delete
    client.get_job.assert_not_called()


def test_remove_training_set_items_from_dataset_polls():
    client = NucleusClient(api_key="test")
    # Removing a whole dataset streams out via a job — the response carries a job_id to poll.
    client.connection.make_request = MagicMock(
        return_value={
            "training_set_id": "ts_1",
            "status": "building",
            "job_id": "job_rm",
        }
    )
    client.get_job = MagicMock()
    client.remove_training_set_items("ts_1", dataset_ids=["ds_1"])
    args = client.connection.make_request.call_args[0]
    assert args[0] == {"dataset_ids": ["ds_1"]}
    assert args[2] is requests.delete
    client.get_job.assert_called_once_with("job_rm")


def test_remove_training_set_items_requires_a_source():
    client = NucleusClient(api_key="test")
    with pytest.raises(ValueError, match="at least one"):
        client.remove_training_set_items("ts_1")


def test_list_training_set_items_paging_query_string():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value={"item_ids": ["di_1", "di_2"], "total": 10}
    )
    page = client.list_training_set_items("ts_1", limit=2, offset=4)
    client.connection.get.assert_called_once_with(
        "trainingSets/ts_1/items?limit=2&offset=4"
    )
    assert page.item_ids == ["di_1", "di_2"]
    assert page.total == 10


def test_list_training_set_items_no_paging_params():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value={"item_ids": [], "total": 0}
    )
    client.list_training_set_items("ts_1")
    client.connection.get.assert_called_once_with("trainingSets/ts_1/items")


# --------------------------------------------------------------------------- #
# Export / download
# --------------------------------------------------------------------------- #
def _export_record(i, *, pointcloud=False):
    """One backend export record (matches the shared export contract)."""
    return {
        "dataset_item_id": f"di_{i}",
        "dataset_id": "ds_1",
        "reference_id": f"ref_{i}",
        "metadata": {"k": i},
        "image_location": None if pointcloud else f"https://x/{i}.jpg",
        "pointcloud_location": f"https://x/{i}.json" if pointcloud else None,
        "width": None if pointcloud else 10,
        "height": None if pointcloud else 20,
    }


def test_export_training_set_items_pages_until_total():
    from nucleus.dataset_item import DatasetItem

    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        side_effect=[
            {"items": [_export_record(0), _export_record(1)], "total": 3},
            {"items": [_export_record(2)], "total": 3},
        ]
    )
    items = client.export_training_set_items("ts_1", limit=2)
    assert [call[0][0] for call in client.connection.get.call_args_list] == [
        "trainingSets/ts_1/export?limit=2&offset=0",
        "trainingSets/ts_1/export?limit=2&offset=2",
    ]
    assert len(items) == 3
    assert all(isinstance(it, DatasetItem) for it in items)
    assert items[0].dataset_item_id == "di_0"
    assert items[0].reference_id == "ref_0"
    assert items[0].image_location == "https://x/0.jpg"
    assert items[0].metadata == {"k": 0}
    assert items[0].width == 10
    assert items[0].height == 20


def test_export_training_set_items_single_page_stops():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value={"items": [_export_record(0)], "total": 1}
    )
    items = client.export_training_set_items("ts_1", limit=1000)
    client.connection.get.assert_called_once_with(
        "trainingSets/ts_1/export?limit=1000&offset=0"
    )
    assert len(items) == 1


def test_export_to_file_writes_jsonl_roundtrip(tmp_path):
    import json

    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value={
            "items": [
                _export_record(1),
                _export_record(2, pointcloud=True),  # pointcloud member
            ],
            "total": 2,
        }
    )
    ts = TrainingSet.from_json(_TRAINING_SET_ROW, client)
    path = tmp_path / "nested" / "export.jsonl"
    count = ts.export_to_file(str(path))

    assert count == 2
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    for row in rows:
        assert "dataset_item_id" in row
        assert "dataset_id" in row
        assert "metadata" in row
    assert rows[0]["dataset_item_id"] == "di_1"
    assert rows[0]["dataset_id"] == "ds_1"
    # The pointcloud member round-trips faithfully (would raise via to_json()).
    assert rows[1]["pointcloud_location"] == "https://x/2.json"
    assert rows[1]["image_location"] is None


def test_download_items_streams_media_to_directory(tmp_path, monkeypatch):
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value={
            "items": [_export_record(1), _export_record(2)],
            "total": 2,
        }
    )
    ts = TrainingSet.from_json(_TRAINING_SET_ROW, client)

    fake_response = MagicMock()
    fake_response.iter_content.return_value = [b"fake-bytes"]
    fake_response.raise_for_status.return_value = None
    context = MagicMock()
    context.__enter__.return_value = fake_response
    fake_get = MagicMock(return_value=context)
    monkeypatch.setattr("nucleus.training_set.requests.get", fake_get)

    count = ts.download_items(str(tmp_path), progress=False)

    assert count == 2
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["ref_1.jpg", "ref_2.jpg"]
    assert (tmp_path / "ref_1.jpg").read_bytes() == b"fake-bytes"
    # No leftover .part temp files.
    assert not any(p.name.endswith(".part") for p in tmp_path.iterdir())


def test_download_items_media_less_record_raises_on_hydration(
    tmp_path, monkeypatch
):
    client = NucleusClient(api_key="test")
    # A member with neither image nor pointcloud location.
    record = {**_export_record(1), "image_location": None}
    record["reference_id"] = "ref_nomedia"
    client.connection.get = MagicMock(
        return_value={"items": [record], "total": 1}
    )
    ts = TrainingSet.from_json(_TRAINING_SET_ROW, client)
    fake_get = MagicMock()
    monkeypatch.setattr("nucleus.training_set.requests.get", fake_get)

    # download_items pages export_items(), which hydrates each record into a
    # DatasetItem; DatasetItem asserts "exactly one media location", so a
    # media-less record raises before any download is attempted.
    with pytest.raises(AssertionError):
        ts.download_items(str(tmp_path), progress=False)
    fake_get.assert_not_called()


# --------------------------------------------------------------------------- #
# Instance methods delegate to the client
# --------------------------------------------------------------------------- #
def test_training_set_instance_methods_delegate_to_client():
    client = MagicMock(spec=NucleusClient)
    ts = TrainingSet(id="ts_1", name="ts", _client=client)

    client.update_training_set.return_value = TrainingSet(
        id="ts_1", name="renamed", _client=client
    )
    ts.update(name="renamed")
    client.update_training_set.assert_called_once_with(
        "ts_1", name="renamed", description=None, metadata=None
    )
    assert ts.name == "renamed"

    ts.delete()
    client.delete_training_set.assert_called_once_with("ts_1")

    ts.items(limit=5)
    client.list_training_set_items.assert_called_once_with(
        "ts_1", limit=5, offset=None
    )

    client.export_training_set_items.return_value = ["item"]
    result = ts.export_items(limit=50)
    client.export_training_set_items.assert_called_once_with("ts_1", limit=50)
    assert result == ["item"]

    client.get_training_set.return_value = TrainingSet(
        id="ts_1", name="renamed", _client=client
    )
    ts.remove_items(["di_1"])
    client.remove_training_set_items.assert_called_once_with(
        "ts_1",
        item_ids=["di_1"],
        items=None,
        slice_id=None,
        dataset_id=None,
        slice_ids=None,
        dataset_ids=None,
        training_set_ids=None,
        wait_for_completion=True,
        verbose=True,
    )

    ts.new_version(removed_item_ids=["di_9"], bump_type="major")
    _, kwargs = client.create_training_set_version.call_args
    assert client.create_training_set_version.call_args[0] == ("ts_1",)
    assert kwargs["removed_item_ids"] == ["di_9"]
    assert kwargs["bump_type"] == "major"


def test_training_set_without_client_raises():
    ts = TrainingSet(id="ts_1", name="ts")
    with pytest.raises(RuntimeError, match="no client"):
        ts.refresh()
