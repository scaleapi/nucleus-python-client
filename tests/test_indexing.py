from nucleus.job import AsyncJob
import pytest

from .helpers import (
    TEST_INDEX_EMBEDDINGS_FILE,
    TEST_IMG_URLS,
    TEST_DATASET_NAME,
    reference_id_from_url,
)

from nucleus import DatasetItem

from nucleus.constants import (
    ERROR_PAYLOAD,
    JOB_ID_KEY,
    MESSAGE_KEY,
    STATUS_KEY,
)


@pytest.fixture()
def dataset(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds_items = []
    for url in TEST_IMG_URLS:
        ds_items.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
            )
        )

    response = ds.append(ds_items)
    assert ERROR_PAYLOAD not in response.json()
    yield ds

    response = CLIENT.delete_dataset(ds.id)
    assert response == {"message": "Beginning dataset deletion..."}


@pytest.mark.integration
def test_index_integration(dataset):
    signed_embeddings_url = TEST_INDEX_EMBEDDINGS_FILE
    job = dataset.create_custom_index([signed_embeddings_url], embedding_dim=3)
    assert job.job_id
    assert job.job_last_known_status
    assert job.job_type
    assert job.job_creation_time

    job_status_response = job.status()
    assert STATUS_KEY in job_status_response
    assert JOB_ID_KEY in job_status_response
    assert MESSAGE_KEY in job_status_response

    job.sleep_until_complete()
    assert job.job_last_known_status == "Completed"


@pytest.mark.skip(reason="Times out consistently")
def test_generate_image_index_integration(dataset):
    job = dataset.create_image_index()
    job.sleep_until_complete()
    job.status()
    assert job.job_last_known_status == "Completed"
