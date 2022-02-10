# Implements the same interface as a nucleus.connection
import time

import requests

from ..constants import DEFAULT_NETWORK_TIMEOUT_SEC
from ..errors import NucleusAPIError
from ..logger import logger
from ..retry_strategy import RetryStrategy


class InternalConnection:
    """Wrapper of HTTP requests to an internal Scale Deploy endpoint.
    Useful if you have a self-hosted version of Deploy."""

    def __init__(self, endpoint, user_id):
        self.endpoint = endpoint
        self.user_id = user_id

    def __repr__(self):
        return f"InternalConnection(user_id='{self.user_id}', endpoint='{self.endpoint}')"

    def __eq__(self, other):
        return (
            self.endpoint == other.endpoint and self.user_id == other.user_id
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
        endpoint = f"{self.endpoint}/{route}"

        logger.info("Make request to %s", endpoint)

        for retry_wait_time in RetryStrategy.sleep_times:
            response = requests_command(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                auth=(self.user_id, ""),  # the only thing that differs hmm
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
