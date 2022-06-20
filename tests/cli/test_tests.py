from unittest import mock

import pytest

from cli.tests import describe_test, list_tests, tests


def test_invoke_tests(runner):
    with mock.patch("cli.tests.list_tests"):
        result = runner.invoke(tests)
    assert result.exception is None
    assert result.exit_code == 0


def test_invoke_tests_web(runner):
    with mock.patch("click.launch"):
        result = runner.invoke(tests, ["--web"])
    assert result.exception is None
    assert result.exit_code == 0


def test_invoke_list_tests(runner, scenario_test):
    result = runner.invoke(list_tests)
    assert result.exception is None
    assert result.exit_code == 0
    assert scenario_test.id in result.output


def test_invoke_describe_test(runner, scenario_test):
    result = runner.invoke(describe_test, [scenario_test.id])
    assert result.exception is None
    assert result.exit_code == 0
    assert scenario_test.id in result.output


@pytest.skip(reason="Errors out on master")
def test_invoke_describe_test_all(runner, scenario_test):
    result = runner.invoke(describe_test, ["--all"])
    assert result.exception is None
    assert result.exit_code == 0
