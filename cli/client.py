import functools

from rich.console import Console


@functools.lru_cache()
def init_client():
    console = Console()
    with console.status("Initializing client"):
        import os

        import nucleus

        api_key = os.environ.get("NUCLEUS_API_KEY", None)
        if api_key:
            client = nucleus.NucleusClient(api_key)
        else:
            raise RuntimeError("No NUCLEUS_API_KEY set")
        return client
