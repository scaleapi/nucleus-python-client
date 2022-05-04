import pytest

try:
    import shapely
except ModuleNotFoundError:
    pytest.skip(
        "Shapely not installed, skipping (install with poetry install -E shapely)",
        allow_module_level=True,
    )
