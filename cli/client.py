import functools
import os

import nucleus


@functools.lru_cache()
def init_client():
    api_key = os.environ.get("NUCLEUS_API_KEY", None)
    if api_key:
        client = nucleus.NucleusClient(api_key)
    else:
        raise RuntimeError("No NUCLEUS_API_KEY set")
    return client
