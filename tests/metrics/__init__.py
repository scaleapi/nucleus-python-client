import pytest

try:
    import rasterio
    import shapely
except ModuleNotFoundError:
    pytest.skip(
        "Shapely or rasterio not installed, skipping (install with poetry install -E metrics)",
        allow_module_level=True,
    )
