import datetime
import requests
from nucleus.constants import (
    JOB_CREATION_TIME_KEY,
    JOB_LAST_KNOWN_STATUS_KEY,
    JOB_TYPE_KEY,
)
from nucleus.job import AsyncJob


def entropy(name, model_runs, client):
    assert (
        len({model_run.dataset_id for model_run in model_runs}) == 1
    ), f"Model runs have conflicting dataset ids: {model_runs}"
    model_run_ids = [model_run.model_run_id for model_run in model_runs]
    dataset_id = model_runs[0].dataset_id
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
