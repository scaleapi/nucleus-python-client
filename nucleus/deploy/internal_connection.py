# Implements the same interface as a nucleus.connection
import time

import requests


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

        raise NotImplementedError

    def handle_bad_response(
        self,
        endpoint,
        requests_command,
        requests_response=None,
        aiohttp_response=None,
    ):
        raise NotImplementedError
