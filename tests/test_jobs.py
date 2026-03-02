import pytest

from nucleus import AsyncJob, NucleusClient

from .helpers import TEST_DATASET_ITEMS, TEST_DATASET_NAME


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


@pytest.fixture(scope="module")
def job_from_dataset_upload(CLIENT):
    """Create a job by doing an async dataset upload."""
    ds = CLIENT.create_dataset(TEST_DATASET_NAME + " job test", is_scene=False)
    try:
        job = ds.append(TEST_DATASET_ITEMS, asynchronous=True)
        job.sleep_until_complete()
        yield job
    finally:
        CLIENT.delete_dataset(ds.id)


@pytest.mark.integration
def test_job_listing(CLIENT):
    """Test that list_jobs returns results."""
    jobs = CLIENT.list_jobs()
    assert isinstance(jobs, list)
    # Just verify the API works and returns AsyncJob objects
    if len(jobs) > 0:
        assert hasattr(jobs[0], "job_id")


@pytest.mark.integration
def test_job_retrieval(CLIENT, job_from_dataset_upload):
    """Test that we can retrieve a job we created by ID."""
    known_job_id = job_from_dataset_upload.job_id

    fetched_job = CLIENT.get_job(known_job_id)
    assert fetched_job.job_id == known_job_id
