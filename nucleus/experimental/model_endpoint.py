from typing import Dict, Sequence


class ModelEndpoint:
    """
    Represents an endpoint on Hosted Model Inference
    """

    def __init__(self, endpoint_id, client):
        self.endpoint_id = endpoint_id
        self.client = client

    def __str__(self):
        return f"ModelEndpoint <endpoint_id:{self.endpoint_id}>"

    def infer(
        self,
        s3urls: Sequence[str],
    ):
        # TODO for demo
        # Make inference requests to the endpoint,
        # if batches are possible make this aware you can pass batches

        # TODO batches once those are out

        request_ids = {}  # Dict of s3url -> request id

        request_endpoint = (
            f"task_async/{self.endpoint_id}"  # is endpoint_name correct?
        )
        for s3url in s3urls:
            # payload = dict(img_url=s3url)  # TODO format idk
            payload = s3url
            # TODO make these requests in parallel instead of making them serially
            # TODO client currently doesn't have a no-json option
            inference_request = self.client.post(
                payload=payload, route=request_endpoint, use_json=False
            )  # Avoid using json because endpoint expects raw url
            request_ids[s3url] = inference_request["task_id"]
            # make the request to the endpoint (in parallel or something)

        return ModelEndpointAsyncJob(
            self.client,
            request_ids=request_ids,
        )

    def status(self):
        # Makes call to model status endpoint,
        raise NotImplementedError

    def sync_request(self, s3url: str):
        # Makes a single request to the synchronous endpoint
        return self.client.sync_request(self.endpoint_id, s3url)


class ModelEndpointAsyncJob:
    """
    Currently represents a list of async inference requests to a specific endpoint

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
        self.request_ids = request_ids.copy()  # s3url -> task_id
        self.responses = {s3url: None for s3url in request_ids.keys()}

    def poll_endpoints(self):
        """
        Runs one round of polling the endpoint for async task results
        """

        # TODO: replace with batch endpoint, or make requests in parallel
        # TODO use the low level HMI client please
        for s3url, request_id in self.request_ids.items():
            current_response = self.responses[s3url]
            if current_response is None:
                payload = {}
                response = self.client.get(
                    payload, f"task/result/{request_id}"
                )
                print(response)
                if (
                    "result_url" not in response
                ):  # TODO no idea what response looks like as of now
                    continue
                self.responses[s3url] = response["result_url"]

    def is_done(self, poll=True):
        """
        Checks if all the tasks from this round of requests are done, according to
        the internal state of this object.
        Optionally polls the endpoints
        """
        # TODO: make some request to some endpoint
        if poll:
            self.poll_endpoints()
        return all(resp is not None for resp in self.responses.values())

    def get_responses(self):
        if not self.is_done(poll=False):
            raise ValueError("Not all responses are done")
        return self.responses.copy()
