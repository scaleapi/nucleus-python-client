import os

import pytest

from nucleus.dataset import Dataset
from nucleus.errors import NucleusAPIError
from tests.helpers import (
    DATASET_WITH_EMBEDDINGS,
    running_as_nucleus_pytest_user,
)

# TODO: Test delete_autotag once API support for autotag creation is added.


@pytest.mark.skip(
    reason="Skip Temporarily - Need to find issue with long running test (2hrs...)"
)
@pytest.mark.integration
def test_update_autotag(CLIENT):
    if running_as_nucleus_pytest_user(CLIENT):
        job = Dataset(DATASET_WITH_EMBEDDINGS, CLIENT).update_autotag(
            "tag_c8jwr0rpy1w00e134an0"
        )
        job.sleep_until_complete()
        status = job.status()
        assert status["status"] == "Completed"


def test_dataset_export_autotag_training_items(CLIENT):
    # This test can only run for the test user who has an indexed dataset.
    # TODO: if/when we can create autotags via api, create one instead.
    if running_as_nucleus_pytest_user(CLIENT):
        dataset = CLIENT.get_dataset(DATASET_WITH_EMBEDDINGS)

        with pytest.raises(NucleusAPIError) as api_error:
            dataset.autotag_training_items(autotag_name="NONSENSE_GARBAGE")
        assert (
            f"The autotag NONSENSE_GARBAGE was not found in dataset {DATASET_WITH_EMBEDDINGS}"
            in str(api_error.value)
        )

        items = dataset.autotag_training_items(autotag_name="PytestTestTag")

        assert "autotagPositiveTrainingItems" in items
        assert "autotag" in items

        autotagTrainingItems = items["autotagPositiveTrainingItems"]
        autotag = items["autotag"]

        assert len(autotagTrainingItems) > 0
        for item in autotagTrainingItems:
            for column in ["ref_id"]:
                assert column in item

        for column in ["id", "name", "status", "autotag_level"]:
            assert column in autotag


def test_export_embeddings(CLIENT):
    if running_as_nucleus_pytest_user(CLIENT):
        embeddings = Dataset(
            DATASET_WITH_EMBEDDINGS, CLIENT
        ).export_embeddings()
        assert "embedding_vector" in embeddings[0]
        assert "reference_id" in embeddings[0]


# TODO(drake): investigate why this only flakes in circleci
@pytest.mark.skip(reason="Flaky test")
def test_dataset_export_autotag_tagged_items(CLIENT):
    # This test can only run for the test user who has an indexed dataset.
    # TODO: if/when we can create autotags via api, create one instead.
    if running_as_nucleus_pytest_user(CLIENT):
        dataset = CLIENT.get_dataset(DATASET_WITH_EMBEDDINGS)

        with pytest.raises(NucleusAPIError) as api_error:
            dataset.autotag_items(autotag_name="NONSENSE_GARBAGE")
        assert (
            f"The autotag NONSENSE_GARBAGE was not found in dataset {DATASET_WITH_EMBEDDINGS}"
            in str(api_error.value)
        )

        items = dataset.autotag_items(autotag_name="PytestTestTag")

        assert "autotagItems" in items
        assert "autotag" in items

        autotagItems = items["autotagItems"]
        autotag = items["autotag"]

        assert len(autotagItems) > 0
        for item in autotagItems:
            for column in ["ref_id", "score"]:
                assert column in item

        for column in ["id", "name", "status", "autotag_level"]:
            assert column in autotag


def test_export_slice_embeddings(CLIENT):
    if running_as_nucleus_pytest_user(CLIENT):
        test_slice = CLIENT.get_slice("slc_c8jwtmj372xg07g9v3k0")
        embeddings = test_slice.export_embeddings()
        assert "embedding_vector" in embeddings[0]
        assert "reference_id" in embeddings[0]


def test_get_autotag_refinement_metrics(CLIENT):
    if running_as_nucleus_pytest_user(CLIENT):
        response = CLIENT.get_autotag_refinement_metrics(
            "tag_c8jwr0rpy1w00e134an0"
        )
        assert response["total_refinement_steps"] >= 0
        assert response["average_positives_selected_per_refinement"] >= 0
