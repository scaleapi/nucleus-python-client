import os

import pytest

from nucleus.dataset import Dataset
from nucleus.errors import NucleusAPIError
from tests.helpers import DATASET_WITH_AUTOTAG, running_as_nucleus_pytest_user

# TODO: Test delete_autotag once API support for autotag creation is added.


@pytest.mark.integration
def test_update_autotag(CLIENT):
    if running_as_nucleus_pytest_user(CLIENT):
        job = Dataset(DATASET_WITH_AUTOTAG, CLIENT).update_autotag(
            "tag_c5jwvzzde8c00604mkx0"
        )
        job.sleep_until_complete()
        status = job.status()
        assert status["status"] == "Completed"


def test_dataset_export_autotag_training_items(CLIENT):
    # This test can only run for the test user who has an indexed dataset.
    # TODO: if/when we can create autotags via api, create one instead.
    if running_as_nucleus_pytest_user(CLIENT):
        dataset = CLIENT.get_dataset(DATASET_WITH_AUTOTAG)

        with pytest.raises(NucleusAPIError) as api_error:
            dataset.autotag_training_items(autotag_name="NONSENSE_GARBAGE")
        assert (
            f"The autotag NONSENSE_GARBAGE was not found in dataset {DATASET_WITH_AUTOTAG}"
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
        embeddings = Dataset(DATASET_WITH_AUTOTAG, CLIENT).export_embeddings()
        assert "embedding_vector" in embeddings[0]
        assert "reference_id" in embeddings[0]


def test_dataset_export_autotag_tagged_items(CLIENT):
    # This test can only run for the test user who has an indexed dataset.
    # TODO: if/when we can create autotags via api, create one instead.
    if running_as_nucleus_pytest_user(CLIENT):
        dataset = CLIENT.get_dataset(DATASET_WITH_AUTOTAG)

        with pytest.raises(NucleusAPIError) as api_error:
            dataset.autotag_items(autotag_name="NONSENSE_GARBAGE")
        assert (
            f"The autotag NONSENSE_GARBAGE was not found in dataset {DATASET_WITH_AUTOTAG}"
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
        test_slice = CLIENT.get_slice("slc_c6kcx5mrzr7g0c9d8cng")
        embeddings = test_slice.export_embeddings()
        assert "embedding_vector" in embeddings[0]
        assert "reference_id" in embeddings[0]
