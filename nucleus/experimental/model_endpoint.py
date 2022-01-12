from typing import Sequence, Dict

import cloudpickle
import requests
import smart_open
from boto3 import Session

import nucleus
from nucleus import Dataset, DatasetItem, NucleusClient
from nucleus.experimental.utils import _nucleus_ds_to_s3url_list


class ModelEndpoint:
    """
    Represents an endpoint on Hosted Model Inference
    TODO remove all mentions of Nucleus objects, i.e. Dataset, DatasetItem, etc.
    """
    def __init__(self, endpoint_id, client):
        self.endpoint_id = endpoint_id
        self.client = client

    def __str__(self):
        return f"ModelEndpoint <endpoint_id:{self.endpoint_id}>"

    def create_run_job(self, dataset: Dataset):
        # TODO: for demo

        s3urls, s3url_to_dataset_map = _nucleus_ds_to_s3url_list(dataset)

        # TODO: pass s3URLs to some run job creation endpoint

        return self._infer(s3urls, s3url_to_dataset_map)

        # Try to upload resulting predictions to nucleus

    def _infer(self, s3urls: Sequence[str], s3url_to_dataset_map: Dict[str, DatasetItem]):
        # TODO for demo
        # Make inference requests to the endpoint,
        # if batches are possible make this aware you can pass batches

        # TODO batches once those are out

        request_ids = {}  # Dict of s3url -> request id

        request_endpoint = f"task_async/{self.endpoint_id}"  # is endpoint_name correct?
        for s3url in s3urls:
            # payload = dict(img_url=s3url)  # TODO format idk
            payload = s3url
            # TODO make these requests in parallel instead of making them serially
            # TODO client currently doesn't have a no-json option
            inference_request = self.client.post(payload=payload, route=request_endpoint, use_json=False)  # Avoid using json because endpoint expects raw url
            request_ids[s3url] = inference_request['task_id']
            # make the request to the endpoint (in parallel or something)

        return ModelEndpointAsyncJob(self.client, request_ids=request_ids, s3url_to_dataset_map=s3url_to_dataset_map)

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

    """
    def __init__(self, client, request_ids: Dict[str, str], s3url_to_dataset_map: Dict[str, DatasetItem]):

        self.client = client
        self.request_ids = request_ids.copy()  # s3url -> task_id
        self.responses = {s3url: None for s3url in request_ids.keys()}
        self.s3url_to_dataset_map = s3url_to_dataset_map

    def poll_endpoints(self):
        """
        Runs one round of polling the endpoint for async task results
        """

        # TODO: make requests in parallel
        for s3url, request_id in self.request_ids.items():
            current_response = self.responses[s3url]
            if current_response is None:
                payload = {}
                response = self.client.get(payload, f"task/result/{request_id}")
                print(response)
                if "result_url" not in response:  # TODO no idea what response looks like as of now
                    continue
                else:
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
        return all([resp is not None for resp in self.responses.values()])

    def get_responses(self):
        if not self.is_done(poll=False):
            raise ValueError("Not all responses are done")
        return self.responses.copy()

    def upload_responses_to_nucleus(self, nucleus_client: NucleusClient, dataset: Dataset, model_run_name: str, model=None, model_name=None, model_ref_id=None):
        """

        """
        # TODO it seems weird to pass in a Dataset again, since this AsyncJob knows about the dataset items themselves
        # TODO untested
        # TODO do we even want to have client-side upload?

        if not self.is_done(poll=False):
            raise ValueError("Not all responses are done")

        # Create a Nucleus model if we don't have one
        if model is None:
            assert model_name is not None and model_ref_id is not None, "If you don't pass a nucleus model you better pass a model name and reference id"
            model = nucleus_client.add_model(name=model_name, reference_id=model_ref_id)

        # Create a Nucleus model run
        model_run = model.create_run(name=model_run_name, dataset=dataset, predictions=[])
        prediction_items = []
        for s3url, dataset_item in self.s3url_to_dataset_map.items():
            item_link = self.responses[s3url]
            print(f"item_link={item_link}")
            # e.g. s3://scale-ml/tmp/hosted-model-inference-outputs/a224499e-50ac-4b08-ad0c-c18e74c14184.pkl
            kwargs = {
                "transport_params": {"session": Session(profile_name="ml-worker")}
            }

            with smart_open.open(item_link, "rb", **kwargs) as bundle_pkl:
                inference_result = cloudpickle.load(bundle_pkl)
                ref_id = dataset_item.reference_id
                for box in inference_result:
                    # TODO assuming box is a list of (x, y, w, h, label). This is almost certainly not the case.
                    # Also, label is probably returned as an integer instead of a label that makes semantic sense
                    pred_item = nucleus.BoxPrediction(
                        label=box["label"],
                        x=box["left"],
                        y=box["top"],
                        width=box["width"],
                        height=box["height"],
                        reference_id=ref_id
                    )
                    prediction_items.append(pred_item)

        job = model_run.predict(prediction_items, asynchronous=True)
        job.sleep_until_complete()
        job.errors()


class ModelBundle:
    """
    Represents a ModelBundle
    """

    def __init__(self, name):
        self.name = name