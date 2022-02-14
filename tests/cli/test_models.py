from click.testing import CliRunner

from cli.models import list_models, models


def test_invoke_models(runner, module_scope_models):
    result = runner.invoke(models)  # type: ignore
    assert result.exception is None
    assert result.exit_code == 0
    for model in module_scope_models:
        assert model.id in result.output


def test_invoke_models_list(runner, module_scope_models):
    result = runner.invoke(list_models)  # type: ignore
    assert result.exception is None
    assert result.exit_code == 0
    for model in module_scope_models:
        assert model.id in result.output
