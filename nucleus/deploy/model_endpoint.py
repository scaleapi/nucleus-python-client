from typing import Dict, Optional, Sequence

TASK_PENDING_STATE = "PENDING"
TASK_SUCCESS_STATE = "SUCCESS"
TASK_FAILURE_STATE = "FAILURE"


class AsyncModelEndpoint:
    """
    A higher level abstraction for a Model Endpoint.
    """

    def __init__(self, endpoint_id: str, client):
        """
        Parameters:
            endpoint_id: The unique name of the ModelEndpoint
            client: A DeployClient object
        """
        self.endpoint_id = endpoint_id
        self.client = client

    def __str__(self):
        return f"ModelEndpoint <endpoint_id:{self.endpoint_id}>"

    def predict(
        self,
        urls: Sequence[str],
    ) -> "AsyncModelEndpointResponse":
        """
        Runs inference on the data items specified by urls. Returns a AsyncModelEndpointResponse.

        Parameters:
            urls: The list of URLs that should have inference run on them. Supported url formats are http(s)://, s3://.

        Returns:
            an AsyncModelEndpointResponse keeping track of the inference requests made
        """
        # Make inference requests to the endpoint,
        # if batches are possible make this aware you can pass batches
        # TODO add batch support once those are out

        request_ids = {}  # Dict of url -> request id

        for url in urls:
            # TODO make these requests in parallel instead of making them serially
            inference_request = self.client.async_request(
                endpoint_id=self.endpoint_id,
                url=url,
            )
            request_ids[url] = inference_request
            # make the request to the endpoint (in parallel or something)

        return AsyncModelEndpointResponse(
            self.client,
            request_ids=request_ids,
        )

    def status(self):
        """Gets the status of the ModelEndpoint.
        TODO this functionality currently does not exist on the server.
        """
        raise NotImplementedError

    async def async_request(self, url: str) -> str:
        """
        Makes an async request to the endpoint. Polls the endpoint under the hood, but provides async/await semantics
        on top.

        Parameters:
            url: A url that points to a file containing model input.
                Must be accessible by Scale Deploy, hence it needs to either be public or a signedURL.

        Returns:
            A signedUrl that contains a cloudpickled Python object, the result of running inference on the model input
            Example output:
                `https://foo.s3.us-west-2.amazonaws.com/bar/baz/qux?xyzzy`
        """
        # TODO implement some lower level async stuff inside client library (some asyncio client)
        raise NotImplementedError


class AsyncModelEndpointResponse:
    """

    Currently represents a list of async inference requests to a specific endpoint. Keeps track of the requests made,
    and gives a way to poll for their status.

    Invariant: set keys for self.request_ids and self.responses are equal

    idk about this abstraction tbh, could use a redesign maybe?

    Also batch inference sort of removes the need for much of the complication in here

    """

    def __init__(
        self,
        client,
        request_ids: Dict[str, str],
    ):

        self.client = client
        self.request_ids = request_ids.copy()  # url -> task_id
        self.responses: Dict[str, Optional[str]] = {
            url: None for url in request_ids.keys()
        }
        # celery task statuses
        self.statuses: Dict[str, Optional[str]] = {
            url: TASK_PENDING_STATE for url in request_ids.keys()
        }

    def poll_endpoints(self):
        """
        Runs one round of polling the endpoint for async task results
        """

        # TODO: replace with batch endpoint, or make requests in parallel
        for url, request_id in self.request_ids.items():
            current_state = self.statuses[url]
            if current_state == TASK_PENDING_STATE:
                response = self.client.get_async_response(request_id)
                print(response)
                if "state" in response:
                    self.statuses[url] = response["state"]
                if "result_url" in response:
                    self.responses[url] = response["result_url"]

    def is_done(self, poll=True) -> bool:
        """
        Checks if all the tasks from this round of requests are done, according to
        the internal state of this object.
        Optionally polls the endpoints to pick up new tasks that may have finished.
        """
        # TODO: make some request to some endpoint
        if poll:
            self.poll_endpoints()
        return all(
            resp != TASK_PENDING_STATE for resp in self.responses.values()
        )

    def get_responses(self) -> Dict[str, Optional[str]]:
        if not self.is_done(poll=False):
            raise ValueError("Not all responses are done")
        return self.responses.copy()

    async def wait(self):
        """
        Waits for inference results to complete. Provides async/await semantics, but under the hood does polling.
        TODO: we'd need to implement some lower level asyncio request code
        """
        raise NotImplementedError
