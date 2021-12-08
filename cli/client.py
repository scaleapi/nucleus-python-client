import os

import nucleus


def compose_client():
    # TODO: Use env var!
    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])
    return client
