import functools

from rich.console import Console


@functools.lru_cache()
def init_client():
    console = Console()
    with console.status("Initializing client"):
        import os

        import nucleus

        api_key = os.environ.get("NUCLEUS_API_KEY", None)
        limited_access_key = os.environ.get("NUCLEUS_LIMITED_ACCESS_KEY", None)
        if api_key or limited_access_key:
            client = nucleus.NucleusClient(api_key=api_key, limited_access_key=limited_access_key)
        else:
            raise RuntimeError(
                "Set NUCLEUS_API_KEY or NUCLEUS_LIMITED_ACCESS_KEY"
            )
        return client
