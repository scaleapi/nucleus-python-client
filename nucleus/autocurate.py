"""Compute active learning metrics on your predictions.

For more details on usage see the example colab in scripts/autocurate_bdd.ipynb
"""


import datetime

import requests

from nucleus.constants import (
    JOB_CREATION_TIME_KEY,
    JOB_LAST_KNOWN_STATUS_KEY,
    JOB_TYPE_KEY,
)
from nucleus.job import AsyncJob


def entropy(name, model_run, client):
    """Computes the mean entropy across all predictions for each image."""
    model_run_ids = [model_run.model_run_id]
    dataset_id = model_run.dataset_id
    response = client.make_request(
        payload={"modelRunIds": model_run_ids},
        route=f"autocurate/{dataset_id}/single_model_entropy/{name}",
        requests_command=requests.post,
    )
    # TODO: the response should already have the below three fields populated
    response[JOB_LAST_KNOWN_STATUS_KEY] = "Started"
    response[JOB_TYPE_KEY] = "autocurateEntropy"
    response[JOB_CREATION_TIME_KEY] = (
        datetime.datetime.now().isoformat("T", "milliseconds") + "Z"
    )
    job = AsyncJob.from_json(response, client)
    return job
