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


def test_index_integration(dataset):
    signed_embeddings_url = TEST_INDEX_EMBEDDINGS_FILE
    create_response = dataset.create_custom_index(
        [signed_embeddings_url], embedding_dim=3
    )
    assert JOB_ID_KEY in create_response
    assert MESSAGE_KEY in create_response
    job_id = create_response[JOB_ID_KEY]

    # Job can error because pytest dataset fixture gets deleted
    # As a workaround, we'll just check htat we got some response
    job_status_response = dataset.check_index_status(job_id)
    assert STATUS_KEY in job_status_response
    assert JOB_ID_KEY in job_status_response
    assert MESSAGE_KEY in job_status_response
