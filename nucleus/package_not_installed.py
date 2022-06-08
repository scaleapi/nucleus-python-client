import sys


class PackageNotInstalled:
    def __init__(self, *args, **kwargs):
        self.raise_error_msg()

    def __getattr__(self, item):
        self.raise_error_msg()

    def raise_error_msg(self):
        """Object to make sure we only raise errors if actually trying to use shapely"""
        if sys.platform.startswith("darwin"):
            platform_specific_msg = (
                "Depending on Python environment used GEOS might need to be installed via "
                "`brew install geos`."
            )
        elif sys.platform.startswith("linux"):
            platform_specific_msg = (
                "Depending on Python environment used GEOS might need to be installed via "
                "system package `libgeos-dev`."
            )
        else:
            platform_specific_msg = "GEOS package will need to be installed see (https://trac.osgeo.org/geos/)"
        raise ModuleNotFoundError(
            f"Module 'shapely' not found. Install optionally with `scale-nucleus[metrics]` or when developing "
            f"`poetry install -E metrics`. {platform_specific_msg}"
        )
