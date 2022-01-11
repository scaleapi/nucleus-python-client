import time

from cli.jobs import jobs, list_jobs
from tests.helpers import get_uuid
from tests.test_dataset import make_dataset_items


def test_invoke_jobs(runner):
    result = runner.invoke(jobs)
    assert result.exit_code == 0


def test_list_jobs(CLIENT, runner):
    dataset = CLIENT.create_dataset(f"[PyTest] Dataset {get_uuid()}")
    items = make_dataset_items()
    dataset.append(items, asynchronous=True)
    result = runner.invoke(list_jobs)
    time.sleep(0.5)
    assert result.exit_code == 0
