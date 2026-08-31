"""Unit tests for benchmarks and benchmark evaluations (no live API)."""

from unittest.mock import MagicMock

import pytest
import requests

from nucleus import (
    AllowedLabelMatch,
    Benchmark,
    EvaluationV2Preset,
    LabelExclusionRule,
    NucleusClient,
    RollupGroup,
)

_BENCHMARK_ROW = {
    "benchmark_id": "bm_1",
    "name": "city-streets",
    "description": "desc",
    "metadata": {"team": "av"},
    "created_by_user_id": "u_1",
    "created_at": "2026-07-10T00:00:00.000Z",
    "item_count": 10,
    "dataset_count": 2,
}

_EVAL_ROW = {
    "id": "evalv2_1",
    "model_run_id": "run_1",
    "dataset_id": "ds_1",
    "benchmark_id": "bm_1",
    "status": "pending",
}


# --------------------------------------------------------------------------- #
# Benchmark CRUD
# --------------------------------------------------------------------------- #
def test_benchmark_from_json_maps_benchmark_id():
    b = Benchmark.from_json(
        {**_BENCHMARK_ROW, "skipped_items_without_ground_truth": 3}
    )
    assert b.id == "bm_1"
    assert b.name == "city-streets"
    assert b.item_count == 10
    assert b.dataset_count == 2
    assert b.skipped_items_without_ground_truth == 3


def test_benchmark_from_json_parses_status():
    assert Benchmark.from_json(_BENCHMARK_ROW).status is None
    assert (
        Benchmark.from_json({**_BENCHMARK_ROW, "status": "building"}).status
        == "building"
    )


def test_benchmark_from_json_parses_lineage_fields():
    b = Benchmark.from_json(
        {
            **_BENCHMARK_ROW,
            "status": "draft",
            "parent_benchmark_id": "bm_parent",
            "version_major": 2,
            "version_minor": 1,
            "version_label": "rc1",
        }
    )
    assert b.status == "draft"
    assert b.parent_benchmark_id == "bm_parent"
    assert b.version_major == 2
    assert b.version_minor == 1
    assert b.version_label == "rc1"
    # Root benchmark: lineage fields absent.
    root = Benchmark.from_json(_BENCHMARK_ROW)
    assert root.parent_benchmark_id is None
    assert root.version_major is None


def _mock_async_create(client, *, benchmark_row=None):
    """Wire up the async create_benchmark flow: 202 {benchmark_id, job_id} ->
    poll the build job -> re-fetch the ready benchmark."""
    client.connection.post = MagicMock(
        return_value={"benchmark_id": "bm_1", "job_id": "job_1"}
    )
    client.get_job = (
        MagicMock()
    )  # .sleep_until_complete() is a no-op MagicMock
    client.get_benchmark = MagicMock(
        return_value=Benchmark.from_json(
            benchmark_row or {**_BENCHMARK_ROW, "status": "ready"}, client
        )
    )
    return client


def test_create_benchmark_from_slice_polls_then_returns_ready():
    client = _mock_async_create(NucleusClient(api_key="test"))
    benchmark = client.create_benchmark(
        "city-streets", description="desc", slice_id="slc_1"
    )
    payload, route = client.connection.post.call_args[0]
    assert route == "benchmarks"
    assert payload == {
        "name": "city-streets",
        "description": "desc",
        "slice_id": "slc_1",
    }
    # Blocks on the build job by default, then fetches the ready benchmark.
    client.get_job.assert_called_once_with("job_1")
    client.get_job.return_value.sleep_until_complete.assert_called_once()
    client.get_benchmark.assert_called_once_with("bm_1")
    assert benchmark.id == "bm_1"
    assert benchmark.status == "ready"


def test_create_benchmark_from_item_ids_and_metadata():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_benchmark(
        "city-streets",
        metadata={"team": "av"},
        item_ids=["di_1", "di_2"],
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["item_ids"] == ["di_1", "di_2"]
    assert payload["metadata"] == {"team": "av"}


def test_create_benchmark_no_wait_returns_building_without_polling():
    client = _mock_async_create(
        NucleusClient(api_key="test"),
        benchmark_row={
            **_BENCHMARK_ROW,
            "status": "building",
            "item_count": 0,
        },
    )
    benchmark = client.create_benchmark(
        "city-streets", slice_id="slc_1", wait_for_completion=False
    )
    client.get_job.assert_not_called()
    client.get_benchmark.assert_called_once_with("bm_1")
    assert benchmark.status == "building"


def test_create_benchmark_requires_at_least_one_member_source():
    client = NucleusClient(api_key="test")
    with pytest.raises(ValueError, match="at least one"):
        client.create_benchmark("b")
    with pytest.raises(ValueError, match="at least one"):
        client.create_benchmark("b", slice_ids=[], dataset_ids=[])


def test_create_benchmark_combines_multiple_sources():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_benchmark(
        "multi",
        item_ids=["di_3"],
        slice_ids=["slc_1", "slc_2"],
        dataset_ids=["ds_1", "ds_2"],
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["item_ids"] == ["di_3"]
    assert payload["slice_ids"] == ["slc_1", "slc_2"]
    assert payload["dataset_ids"] == ["ds_1", "ds_2"]


def test_list_benchmarks():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value=[dict(_BENCHMARK_ROW)])
    benchmarks = client.list_benchmarks()
    client.connection.get.assert_called_once_with("benchmarks")
    assert len(benchmarks) == 1
    assert benchmarks[0].id == "bm_1"


def test_get_benchmark():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(return_value=dict(_BENCHMARK_ROW))
    benchmark = client.get_benchmark("bm_1")
    client.connection.get.assert_called_once_with("benchmarks/bm_1")
    assert benchmark.id == "bm_1"


def test_update_benchmark_sends_only_provided_fields():
    client = NucleusClient(api_key="test")
    client.connection.patch = MagicMock(
        return_value={**_BENCHMARK_ROW, "name": "renamed"}
    )
    benchmark = client.update_benchmark("bm_1", name="renamed")
    payload, route = client.connection.patch.call_args[0]
    assert route == "benchmarks/bm_1"
    assert payload == {"name": "renamed"}
    assert benchmark.name == "renamed"


def test_delete_benchmark():
    client = NucleusClient(api_key="test")
    client.connection.make_request = MagicMock(return_value=MagicMock())
    client.delete_benchmark("bm_1")
    args = client.connection.make_request.call_args[0]
    assert args[1] == "benchmarks/bm_1"
    assert args[2] is requests.delete


# --------------------------------------------------------------------------- #
# Versioning / lineage
# --------------------------------------------------------------------------- #
def test_create_benchmark_version_sends_lineage_payload():
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_benchmark(
        "city-streets-v2",
        parent_benchmark_id="bm_parent",
        bump_type="major",
        removed_item_ids=["di_9"],
        item_ids=["di_1"],
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["parent_benchmark_id"] == "bm_parent"
    assert payload["bump_type"] == "major"
    assert payload["removed_item_ids"] == ["di_9"]
    assert payload["item_ids"] == ["di_1"]


def test_create_benchmark_parent_alone_is_a_valid_source():
    # A pure re-version (parent, no added/removed) is allowed.
    client = _mock_async_create(NucleusClient(api_key="test"))
    client.create_benchmark("reversion", parent_benchmark_id="bm_parent")
    payload = client.connection.post.call_args[0][0]
    assert payload["parent_benchmark_id"] == "bm_parent"


def test_create_benchmark_removed_items_requires_parent():
    client = NucleusClient(api_key="test")
    with pytest.raises(ValueError, match="removed_item_ids"):
        client.create_benchmark(
            "b", item_ids=["di_1"], removed_item_ids=["di_9"]
        )


# --------------------------------------------------------------------------- #
# Draft benchmarks
# --------------------------------------------------------------------------- #
def test_create_draft_empty_starts_no_job():
    client = NucleusClient(api_key="test")
    # Empty draft: server returns a benchmark_id with a null job_id.
    client.connection.post = MagicMock(
        return_value={"benchmark_id": "bm_1", "job_id": None}
    )
    client.get_job = MagicMock()
    client.get_benchmark = MagicMock(
        return_value=Benchmark.from_json(
            {**_BENCHMARK_ROW, "status": "draft", "item_count": 0}, client
        )
    )
    benchmark = client.create_benchmark("draft-bm", draft=True)
    payload = client.connection.post.call_args[0][0]
    assert payload["draft"] is True
    # No sources -> no seed job -> never polled, and no "missing job_id" error.
    client.get_job.assert_not_called()
    assert benchmark.status == "draft"


def test_create_draft_with_sources_seeds_and_polls():
    client = _mock_async_create(
        NucleusClient(api_key="test"),
        benchmark_row={**_BENCHMARK_ROW, "status": "draft"},
    )
    client.create_benchmark("draft-bm", draft=True, item_ids=["di_1"])
    payload = client.connection.post.call_args[0][0]
    assert payload["draft"] is True
    assert payload["item_ids"] == ["di_1"]
    client.get_job.assert_called_once_with("job_1")


def test_add_benchmark_items_posts_sources_and_polls():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(return_value={"job_id": "job_add"})
    client.get_job = MagicMock()
    client.add_benchmark_items("bm_1", item_ids=["di_1"], slice_ids=["slc_1"])
    payload, route = client.connection.post.call_args[0]
    assert route == "benchmarks/bm_1/items"
    assert payload["item_ids"] == ["di_1"]
    assert payload["slice_ids"] == ["slc_1"]
    client.get_job.assert_called_once_with("job_add")


def test_add_benchmark_items_requires_a_source():
    client = NucleusClient(api_key="test")
    with pytest.raises(ValueError, match="at least one"):
        client.add_benchmark_items("bm_1")


def test_remove_benchmark_items_deletes_with_body():
    client = NucleusClient(api_key="test")
    client.connection.make_request = MagicMock(return_value=MagicMock())
    client.remove_benchmark_items("bm_1", ["di_1", "di_2"])
    # NucleusClient.make_request forwards positionally to connection.make_request:
    # (payload, route, requests_command, return_raw_response).
    args = client.connection.make_request.call_args[0]
    assert args[0] == {"item_ids": ["di_1", "di_2"]}
    assert args[1] == "benchmarks/bm_1/items"
    assert args[2] is requests.delete


def test_finalize_benchmark_returns_ready():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(
        return_value={**_BENCHMARK_ROW, "status": "ready"}
    )
    benchmark = client.finalize_benchmark("bm_1")
    payload, route = client.connection.post.call_args[0]
    assert route == "benchmarks/bm_1/finalize"
    assert payload == {}
    assert benchmark.status == "ready"


def test_benchmark_finalize_updates_self_in_place():
    client = NucleusClient(api_key="test")
    client.connection.post = MagicMock(
        return_value={**_BENCHMARK_ROW, "status": "ready"}
    )
    draft = Benchmark.from_json({**_BENCHMARK_ROW, "status": "draft"}, client)
    result = draft.finalize()
    assert result is draft
    assert draft.status == "ready"


def test_list_benchmark_items_paging_query_string():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value={"item_ids": ["di_1", "di_2"], "total": 10}
    )
    page = client.list_benchmark_items("bm_1", limit=2, offset=4)
    client.connection.get.assert_called_once_with(
        "benchmarks/bm_1/items?limit=2&offset=4"
    )
    assert page.item_ids == ["di_1", "di_2"]
    assert page.total == 10


def test_list_benchmark_items_no_paging_params():
    client = NucleusClient(api_key="test")
    client.connection.get = MagicMock(
        return_value={"item_ids": [], "total": 0}
    )
    client.list_benchmark_items("bm_1")
    client.connection.get.assert_called_once_with("benchmarks/bm_1/items")


# --------------------------------------------------------------------------- #
# Benchmark instance methods
# --------------------------------------------------------------------------- #
def test_benchmark_instance_methods_delegate_to_client():
    client = MagicMock(spec=NucleusClient)
    benchmark = Benchmark(id="bm_1", name="b", _client=client)

    client.update_benchmark.return_value = Benchmark(
        id="bm_1", name="renamed", _client=client
    )
    benchmark.update(name="renamed")
    client.update_benchmark.assert_called_once_with(
        "bm_1", name="renamed", description=None, metadata=None
    )
    assert benchmark.name == "renamed"

    benchmark.delete()
    client.delete_benchmark.assert_called_once_with("bm_1")

    benchmark.items(limit=5)
    client.list_benchmark_items.assert_called_once_with(
        "bm_1", limit=5, offset=None
    )

    benchmark.create_evaluation_v2("run_1", name="e")
    _, kwargs = client.create_benchmark_evaluation_v2.call_args
    assert client.create_benchmark_evaluation_v2.call_args[0] == (
        "bm_1",
        "run_1",
    )
    assert kwargs["name"] == "e"


def test_benchmark_without_client_raises():
    benchmark = Benchmark(id="bm_1", name="b")
    with pytest.raises(RuntimeError, match="no client"):
        benchmark.refresh()


# --------------------------------------------------------------------------- #
# Benchmark evaluation create
# --------------------------------------------------------------------------- #
def _mock_create_eval(client):
    client.connection.post = MagicMock(
        return_value={"evaluation_id": "evalv2_1"}
    )
    client.connection.get = MagicMock(return_value=dict(_EVAL_ROW))


def test_create_benchmark_evaluation_v2_with_rollup_groups():
    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    evaluation = client.create_benchmark_evaluation_v2(
        "bm_1",
        "run_1",
        name="eval",
        rollup_groups=[RollupGroup("vehicle", ["car", "truck"])],
        exclusion_rules=[
            LabelExclusionRule(
                scope="item", target="prediction", labels=["ignore"]
            )
        ],
    )
    payload, route = client.connection.post.call_args[0]
    assert route == "benchmarks/bm_1/evaluationsV2"
    assert payload["model_run_id"] == "run_1"
    assert payload["name"] == "eval"
    assert payload["rollupGroups"] == [
        {"class_name": "vehicle", "labels": ["car", "truck"]}
    ]
    assert "onlyItemsWithPredictions" not in payload
    client.connection.get.assert_called_once_with("evaluationsV2/evalv2_1")
    assert evaluation.id == "evalv2_1"
    assert evaluation.benchmark_id == "bm_1"


def test_create_benchmark_evaluation_v2_label_config_mutual_exclusion():
    client = NucleusClient(api_key="test")
    with pytest.warns(DeprecationWarning, match="allowed_label_matches"):
        with pytest.raises(ValueError, match="at most one"):
            client.create_benchmark_evaluation_v2(
                "bm_1",
                "run_1",
                rollup_groups=[RollupGroup("vehicle", ["car"])],
                allowed_label_matches=[AllowedLabelMatch("car", "vehicle")],
            )
    with pytest.warns(DeprecationWarning, match="allowed_label_matches"):
        with pytest.raises(ValueError, match="at most one"):
            client.create_benchmark_evaluation_v2(
                "bm_1",
                "run_1",
                rollup_groups=[RollupGroup("vehicle", ["car"])],
                allowed_label_matches_id="alm_1",
            )


def test_create_benchmark_evaluation_v2_preset_seeds_rollup_groups():
    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    preset = EvaluationV2Preset(
        id="prev_1",
        name="p",
        rollup_groups=[RollupGroup("vehicle", ["car"])],
        exclusion_rules=[{"type": "labels", "scope": "item"}],
    )
    client.create_benchmark_evaluation_v2("bm_1", "run_1", preset=preset)
    payload = client.connection.post.call_args[0][0]
    assert payload["rollupGroups"] == [
        {"class_name": "vehicle", "labels": ["car"]}
    ]
    assert payload["exclusionRules"] == [{"type": "labels", "scope": "item"}]
    assert "allowed_label_matches" not in payload


def test_create_benchmark_evaluation_v2_preset_seeds_legacy_matches():
    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    preset = EvaluationV2Preset(
        id="prev_1",
        name="p",
        allowed_label_matches=[AllowedLabelMatch("car", "vehicle")],
    )
    # Seeding from a stored preset does not itself warn — the deprecated
    # kwargs were not passed by the caller.
    client.create_benchmark_evaluation_v2("bm_1", "run_1", preset=preset)
    payload = client.connection.post.call_args[0][0]
    assert payload["allowed_label_matches"] == [
        {"ground_truth_label": "car", "model_prediction_label": "vehicle"}
    ]
    assert "rollupGroups" not in payload


def test_create_benchmark_evaluation_v2_allowed_label_matches_deprecated():
    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    with pytest.warns(DeprecationWarning, match="allowed_label_matches"):
        client.create_benchmark_evaluation_v2(
            "bm_1",
            "run_1",
            allowed_label_matches=[AllowedLabelMatch("car", "vehicle")],
        )
    payload = client.connection.post.call_args[0][0]
    assert payload["allowed_label_matches"] == [
        {"ground_truth_label": "car", "model_prediction_label": "vehicle"}
    ]


def test_create_benchmark_evaluation_v2_allowed_label_matches_id_deprecated():
    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    with pytest.warns(DeprecationWarning, match="allowed_label_matches"):
        client.create_benchmark_evaluation_v2(
            "bm_1",
            "run_1",
            allowed_label_matches_id="alm_1",
        )
    payload = client.connection.post.call_args[0][0]
    assert payload["allowed_label_matches_id"] == "alm_1"


def test_create_benchmark_evaluation_v2_rollup_groups_does_not_warn():
    import warnings

    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        client.create_benchmark_evaluation_v2(
            "bm_1",
            model_id="prj_1",
            rollup_groups=[RollupGroup("vehicle", ["car"])],
        )


def test_create_benchmark_evaluation_v2_model_run_id_deprecated():
    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    with pytest.warns(DeprecationWarning, match="model_run_id"):
        client.create_benchmark_evaluation_v2("bm_1", "run_1")
    payload = client.connection.post.call_args[0][0]
    assert payload["model_run_id"] == "run_1"


def test_create_benchmark_evaluation_v2_model_id_does_not_warn():
    import warnings

    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        client.create_benchmark_evaluation_v2("bm_1", model_id="prj_1")


def test_create_benchmark_evaluation_v2_explicit_args_override_preset():
    client = NucleusClient(api_key="test")
    _mock_create_eval(client)
    preset = EvaluationV2Preset(
        id="prev_1",
        name="p",
        rollup_groups=[RollupGroup("vehicle", ["car"])],
    )
    client.create_benchmark_evaluation_v2(
        "bm_1",
        "run_1",
        rollup_groups=[RollupGroup("person", ["ped"])],
        preset=preset,
    )
    payload = client.connection.post.call_args[0][0]
    assert payload["rollupGroups"] == [
        {"class_name": "person", "labels": ["ped"]}
    ]
