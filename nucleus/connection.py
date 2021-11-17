import logging
import requests
import time

from .constants import DEFAULT_NETWORK_TIMEOUT_SEC
from .errors import NucleusAPIError

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger(requests.packages.urllib3.__package__).setLevel(
    logging.ERROR
)


# TODO: use retry library instead of custom code. Tenacity is one option.
class RetryStrategy:
    statuses = {503, 504}
    sleep_times = [1, 3, 9]


class Connection:
    """Wrapper of HTTP requests to the Nucleus endpoint."""

    def __init__(self, api_key: str, endpoint: str = None):
        self.api_key = api_key
        self.endpoint = endpoint

    def make_request(
        self, payload: dict, route: str, requests_command=requests.post
    ) -> dict:
        """
        Makes a request to Nucleus endpoint and logs a warning if not
        successful.

        :param payload: given payload
        :param route: route for the request
        :param requests_command: requests.post, requests.get, requests.delete
        :return: response JSON
        """
        endpoint = f"{self.endpoint}/{route}"

        logger.info("Make request to %s", endpoint)

        for retry_wait_time in RetryStrategy.sleep_times:
            response = requests_command(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                auth=(self.api_key, ""),
                timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
            )
            logger.info(
                "API request has response code %s", response.status_code
            )
            if response.status_code not in RetryStrategy.statuses:
                break
            time.sleep(retry_wait_time)

        if not response.ok:
            self.handle_bad_response(endpoint, requests_command, response)

        return response.json()

    def handle_bad_response(
        self,
        endpoint,
        requests_command,
        requests_response=None,
        aiohttp_response=None,
    ):
        raise NucleusAPIError(
            endpoint, requests_command, requests_response, aiohttp_response
        )
