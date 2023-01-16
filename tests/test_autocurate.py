import time

import pytest

from nucleus import DatasetItem, autocurate
from nucleus.async_job import AsyncJob
from nucleus.constants import ERROR_PAYLOAD
from nucleus.prediction import BoxPrediction
from tests.helpers import (
    TEST_BOX_PREDICTIONS,
    TEST_DATASET_NAME,
    TEST_IMG_URLS,
    TEST_MODEL_NAME,
    TEST_MODEL_RUN,
    reference_id_from_url,
)


@pytest.fixture()
def model_run(CLIENT):
    ds = CLIENT.create_dataset(TEST_DATASET_NAME)
    ds_items = []
    for url in TEST_IMG_URLS[:2]:
        ds_items.append(
            DatasetItem(
                image_location=url,
                reference_id=reference_id_from_url(url),
            )
        )

    response = ds.append(ds_items)

    assert ERROR_PAYLOAD not in response.json()

    model = CLIENT.create_model(
        name=TEST_MODEL_NAME, reference_id="model_" + str(time.time())
    )

    run = model.create_run(name=TEST_MODEL_RUN, dataset=ds, predictions=[])
    prediction = BoxPrediction(**TEST_BOX_PREDICTIONS[1])
    run.predict(annotations=[prediction])

    yield run

    response = CLIENT.delete_model(model.id)
    assert response == {}


@pytest.mark.integration
@pytest.mark.xfail(reason="Autocurate constantly erroring out.")
def test_autocurate_integration(model_run, CLIENT):
    job = autocurate.entropy("Test Autocurate Integration", model_run, CLIENT)
    job.sleep_until_complete()
    assert job.job_last_known_status == "Completed"
