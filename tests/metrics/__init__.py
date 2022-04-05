import pytest

try:
    import shapely
except ModuleNotFoundError:
    pytest.skip(
        "Skipping metrics tests (to run install with poetry install -E metrics)",
        allow_module_level=True,
    )
