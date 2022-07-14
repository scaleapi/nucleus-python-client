import pytest

try:
    import shapely
except (ModuleNotFoundError, OSError):
    pytest.skip(
        "Skipping metrics tests (to run install with poetry install -E shapely)",
        allow_module_level=True,
    )

# TODO(gunnar): Add Cuboid tests!
