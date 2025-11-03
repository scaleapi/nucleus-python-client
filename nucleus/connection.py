import os
import time
from typing import Optional

import requests

from .constants import DEFAULT_NETWORK_TIMEOUT_SEC
from .errors import NucleusAPIError, NoAPIKey
from .logger import logger
from .retry_strategy import RetryStrategy


class Connection:
    """Wrapper of HTTP requests to the Nucleus endpoint."""

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, extra_headers: Optional[dict] = None):
        self.api_key = api_key
        self.endpoint = endpoint
        self.extra_headers = extra_headers or {}
        # Require at least one auth mechanism: Basic (api_key) or limited access header
        if self.api_key is None and not self.extra_headers.get("x-limited-access-key"):
            raise NoAPIKey()

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
        self,
        payload: dict,
        route: str,
        requests_command=requests.post,
        return_raw_response: bool = False,
    ) -> dict:
        """
        Makes a request to Nucleus endpoint and logs a warning if not
        successful.

        :param payload: given payload
        :param route: route for the request
        :param requests_command: requests.post, requests.get, requests.delete
        :param return_raw_response: return the request's response object entirely
        :return: response JSON
        """
        endpoint = f"{self.endpoint}/{route}"

        logger.info("Make request to %s", endpoint)

        for retry_wait_time in RetryStrategy.sleep_times():
            auth_kwargs = (
                {"auth": (self.api_key, "")} if self.api_key is not None else {}
            )
            response = requests_command(
                endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    **(self.extra_headers or {}),
                },
                timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
                verify=os.environ.get("NUCLEUS_SKIP_SSL_VERIFY", None) is None,
                **auth_kwargs,
            )
            logger.info(
                "API request has response code %s", response.status_code
            )
            if response.status_code not in RetryStrategy.statuses:
                break
            time.sleep(retry_wait_time)

        if not response.ok:
            self.handle_bad_response(endpoint, requests_command, response)

        if return_raw_response:
            return response

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
