import time

import pytest
from click.testing import CliRunner

from cli.datasets import datasets, delete_dataset, list_datasets


def test_invoke_dataset_outputs_list(cli_datasets):
    runner = CliRunner()
    result = runner.invoke(datasets)  # type: ignore
    assert result.exception is None
    for dataset in cli_datasets:
        assert dataset.id in result.output


def test_invoke_dataset_list(cli_datasets):
    runner = CliRunner()
    result = runner.invoke(list_datasets)  # type: ignore
    assert result.exception is None
    for dataset in cli_datasets:
        assert dataset.id in result.output


@pytest.mark.xfail(
    reason="Dataset delete is flaky -> The deleted dataset still shows up."
)
def test_invoke_dataset_delete(CLIENT, cli_datasets):
    dataset_name = "[PyTest] To delete"
    dataset = CLIENT.create_dataset(dataset_name, is_scene=False)
    runner = CliRunner()
    result = runner.invoke(delete_dataset, ["--id", str(dataset.id)], input="DELETE")  # type: ignore
    assert result.exception is None
    assert result.exit_code == 0
    # NOTE: Takes a bit of time to delete -> Sleep
    time.sleep(2)
    list_result = runner.invoke(list_datasets)  # type:ignore
    assert dataset.id not in list_result.output


def test_invoke_dataset_delete_wrong_confirmation(cli_datasets):
    runner = CliRunner()
    dataset_to_not_delete = cli_datasets[0]
    result = runner.invoke(delete_dataset, ["--id", str(dataset_to_not_delete.id)], input="DEL")  # type: ignore
    assert result.exit_code != 0
    list_result = runner.invoke(list_datasets)  # type:ignore
    assert dataset_to_not_delete.id in list_result.output
