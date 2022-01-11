from unittest import mock

from cli.tests import tests, list_tests, describe_test

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

def test_invoke_list_tests(runner, unit_test):
    result = runner.invoke(list_tests)
    assert result.exception is None
    assert result.exit_code == 0
    assert unit_test.id in result.output

def test_invoke_describe_test(runner, unit_test):
    result = runner.invoke(describe_test, [unit_test.id])
    assert result.exception is None
    assert result.exit_code == 0
    assert unit_test.id in result.output

def test_invoke_describe_test_all(runner, unit_test):
    result = runner.invoke(describe_test, ["--all"])
    assert result.exception is None
    assert result.exit_code == 0
