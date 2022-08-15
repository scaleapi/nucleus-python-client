import time
from pathlib import Path

import pytest

from nucleus import AsyncJob, NucleusClient


def test_reprs():
    # Have to define here in order to have access to all relevant objects
    def test_repr(test_object: any):
        assert eval(str(test_object)) == test_object

    client = NucleusClient(api_key="fake_key")
    test_repr(
        AsyncJob(
            client=client,
            job_id="fake_job_id",
            job_last_known_status="fake_job_status",
            job_type="fake_job_type",
            job_creation_time="fake_job_creation_time",
        )
    )


def test_job_listing_and_retrieval(CLIENT):
    jobs = CLIENT.list_jobs()

    if not jobs:
        return

    for job in jobs:
        assert eval(str(job)) == job

    fetch_id = jobs[0].job_id
    fetched_job = CLIENT.get_job(fetch_id)
    assert fetched_job.status() == jobs[0].status()
