import os
from urllib.parse import urljoin


def nucleus_url(sub_path: str):
    nucleus_base = os.environ.get(
        "NUCLEUS_DASHBOARD", "https://dashboard.scale.com/nucleus/"
    )
    extra_params = os.environ.get("NUCLEUS_DASH_PARAMS", "")
    return urljoin(nucleus_base, sub_path.lstrip("/") + extra_params)
