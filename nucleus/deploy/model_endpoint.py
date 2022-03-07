import concurrent.futures
import uuid
from collections import Counter
from typing import Dict, Optional, Sequence

from nucleus.deploy.request_validation import validate_task_request

TASK_PENDING_STATE = "PENDING"
TASK_SUCCESS_STATE = "SUCCESS"
TASK_FAILURE_STATE = "FAILURE"


class EndpointRequest:
    """
    Represents a single request to either a SyncModelEndpoint or AsyncModelEndpoint.
    Parameters:
        url: A url to some file that can be read in to a ModelBundle's predict function. Can be an image, raw text, etc.
        args: A Dictionary with arguments to a ModelBundle's predict function. If the predict function has signature
            predict_fn(foo, bar), then the keys in the dictionary should be 'foo' and 'bar'. Values must be native Python
            objects.
        return_pickled: Whether the output should be a pickled python object, or directly returned serialized json
        request_id: A user-specifiable id for requests.
            Should be unique among EndpointRequests made in the same batch call.
            If one isn't provided the client will generate its own.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        args: Optional[Dict] = None,
        return_pickled: Optional[bool] = True,
        request_id: Optional[str] = None,
    ):
        # TODO: request_id is pretty much here only to support the clientside AsyncModelEndpointBatchResponse
        # so it should be removed when we get proper batch endpoints working.
        validate_task_request(url=url, args=args)
        if request_id is None:
            request_id = str(uuid.uuid4())
        self.url = url
        self.args = args
        self.return_pickled = return_pickled
        self.request_id: str = request_id


class EndpointResponse:
    """
    Represents a response received from a ModelEndpoint.
    Status is a string representing the status of the request, i.e. SUCCESS, FAILURE, or PENDING
    Exactly one of result_url or result will be populated, depending on the value of `return_pickled` in the request.
    result_url is a string that is a url containing the pickled python object from the ModelEndpoint's predict function.
    result is a string that is the serialized return value (in json form) of the ModelEndpoint's predict function.
        Specifically, one can json.loads() the value of result to get the original python object back.
    """

    def __init__(self, status, result_url, result):
        self.status = status
        self.result_url = result_url
        self.result = result

    def __str__(self):
        return f"status: {self.status}, result: {self.result}, result_url: {self.result_url}"


class SyncModelEndpoint:
    def __init__(self, endpoint_id: str, client):
        self.endpoint_id = endpoint_id
        self.client = client

    def __str__(self):
        return f"SyncModelEndpoint <endpoint_id:{self.endpoint_id}>"

    def predict(self, request: EndpointRequest) -> EndpointResponse:
        raw_response = self.client.sync_request(
            self.endpoint_id,
            url=request.url,
            args=request.args,
            return_pickled=request.return_pickled,
        )
        return EndpointResponse(
            status=TASK_SUCCESS_STATE,
            result_url=raw_response.get("result_url", None),
            result=raw_response.get("result", None),
        )

    def status(self):
        # TODO this functionality doesn't exist serverside
        raise NotImplementedError


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
        return f"AsyncModelEndpoint <endpoint_id:{self.endpoint_id}>"

    def predict_batch(
        self, requests: Sequence[EndpointRequest]
    ) -> "AsyncModelEndpointBatchResponse":
        """
        Runs inference on the data items specified by urls. Returns a AsyncModelEndpointResponse.

        Parameters:
            requests: List of EndpointRequests. Request_ids must all be distinct.

        Returns:
            an AsyncModelEndpointResponse keeping track of the inference requests made
        """
        # Make inference requests to the endpoint,
        # if batches are possible make this aware you can pass batches
        # TODO add batch support once those are out

        if len(requests) != len(
            set(request.request_id for request in requests)
        ):
            raise ValueError("Request_ids in a batch must be unique")

        def single_request(request):
            # request has keys url and args

            inner_inference_request = self.client.async_request(
                endpoint_id=self.endpoint_id,
                url=request.url,
                args=request.args,
                return_pickled=request.return_pickled,
            )
            request_key = request.request_id
            return request_key, inner_inference_request

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            urls_to_requests = executor.map(single_request, requests)
            request_ids = dict(urls_to_requests)

        return AsyncModelEndpointBatchResponse(
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


class AsyncModelEndpointBatchResponse:
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
        self.request_ids = (
            request_ids.copy()
        )  # custom request_id (clientside) -> task_id (serverside)
        self.responses: Dict[str, Optional[EndpointResponse]] = {
            req_id: None for req_id in request_ids.keys()
        }
        # celery task statuses
        self.statuses: Dict[str, Optional[str]] = {
            req_id: TASK_PENDING_STATE for req_id in request_ids.keys()
        }

    def poll_endpoints(self):
        """
        Runs one round of polling the endpoint for async task results
        """

        # TODO: replace with batch endpoint, or make requests in parallel

        def single_request(inner_url, inner_task_id):
            if self.statuses[inner_url] != TASK_PENDING_STATE:
                # Skip polling tasks that are completed
                return None
            inner_response = self.client.get_async_response(inner_task_id)
            print("inner response", inner_response)
            return (
                inner_url,
                inner_task_id,
                inner_response.get("state", None),
                inner_response,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            responses = executor.map(
                single_request,
                self.request_ids.keys(),
                self.request_ids.values(),
            )

        for response in responses:
            if response is None:
                continue
            url, _, state, raw_response = response
            if state:
                self.statuses[url] = state
            if raw_response:
                response_object = EndpointResponse(
                    status=raw_response["state"],
                    result_url=raw_response.get("result_url", None),
                    result=raw_response.get("result", None),
                )
                self.responses[url] = response_object

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
            resp != TASK_PENDING_STATE for resp in self.statuses.values()
        )

    def get_responses(self) -> Dict[str, Optional[EndpointResponse]]:
        """
        Returns a dictionary, where each key is the request_id for an EndpointRequest passed in, and the corresponding
        object at that key is the corresponding EndpointResponse.
        """
        if not self.is_done(poll=False):
            raise ValueError("Not all responses are done")
        return self.responses.copy()

    def batch_status(self):
        counter = Counter(self.statuses.values())
        return dict(counter)

    async def wait(self):
        """
        Waits for inference results to complete. Provides async/await semantics, but under the hood does polling.
        TODO: we'd need to implement some lower level asyncio request code
        """
        raise NotImplementedError
