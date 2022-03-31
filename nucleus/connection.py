import time

import requests

from .constants import DEFAULT_NETWORK_TIMEOUT_SEC
from .errors import NucleusAPIError
from .logger import logger
from .retry_strategy import RetryStrategy


class Connection:
    """Wrapper of HTTP requests to the Nucleus endpoint."""

    def __init__(self, api_key: str, endpoint: str = None):
        self.api_key = api_key
        self.endpoint = endpoint

    def __repr__(self):
        return (
            f"Connection(api_key='{self.api_key}', endpoint='{self.endpoint}')"
        )

    def __eq__(self, other):
        return (
            self.api_key == other.api_key and self.endpoint == other.endpoint
        )

    def delete(self, route: str):
        return self.make_request({}, route, requests_command=requests.delete)

    def get(self, route: str):
        return self.make_request({}, route, requests_command=requests.get)

    def post(self, payload: dict, route: str):
        return self.make_request(
            payload, route, requests_command=requests.post
        )

    def put(self, payload: dict, route: str):
        return self.make_request(payload, route, requests_command=requests.put)

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

        for retry_wait_time in RetryStrategy.sleep_times():
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
