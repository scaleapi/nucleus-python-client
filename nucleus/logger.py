import logging

import requests

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger(
    requests.packages.urllib3.__package__  # pylint: disable=no-member
).setLevel(logging.ERROR)
