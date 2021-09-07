import logging
from typing import Any, Callable, Dict, Sequence, Tuple

import cloudpickle
import requests
import smart_open
from boto3 import Session

import nucleus
from nucleus import NucleusClient
from nucleus.dataset import Dataset
from nucleus.dataset_item import DatasetItemType, DatasetItem

# TODO temporary endpoint, will be replaced with some https://api.scale.com/hostedinference/<sub-route>
HOSTED_INFERENCE_ENDPOINT = "http://hostedinference.ml-internal.scale.com:5000"  # TODO this isn't https
DEFAULT_NETWORK_TIMEOUT_SEC = 120

logger = logging.getLogger(__name__)
logging.basicConfig()


class ModelBundle:
    """
    Represents a ModelBundle
    """

    def __init__(self, name):
        self.name = name


class ModelEndpointAsyncJob:
    """
    Currently represents a list of async inference requests to a specific endpoint

    Invariant: set keys for self.request_ids and self.responses are equal

    """
    def __init__(self, request_ids: Dict[str, str], s3url_to_dataset_map: Dict[str, DatasetItem], model_endpoint: "ModelEndpoint"):
        # TODO is it weird for ModelEndpointAsyncJob to know about ModelEndpoint?
        # probably not, but we might not need model_endpoint anyways heh, depending on
        # the format of the Get Task Result url
        self.request_ids = request_ids.copy()  # s3url -> task_id
        self.responses = {s3url: None for s3url in request_ids.keys()}
        self.s3url_to_dataset_map = s3url_to_dataset_map
        self.model_endpoint = model_endpoint  # TODO unused?

    def poll_endpoints(self):
        """
        Runs one round of polling the endpoint for async task results
        """

        # TODO: make requests in parallel
        for s3url, request_id in self.request_ids.items():
            current_response = self.responses[s3url]
            if current_response is None:
                payload = {}
                response = make_hosted_inference_request(payload, f"task/result/{request_id}", requests_command=requests.get)
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

    def upload_responses_to_nucleus(self, nucleus_client: NucleusClient, dataset: Dataset, model=None):
        """

        """
        # TODO untested

        if not self.is_done(poll=False):
            raise ValueError("Not all responses are done")
        # TODO create a nucleus Model object, or take one in as an argument
        if model is None:
            model = nucleus_client.add_model(name="TODO", reference_id="TODO")
        model_run = model.create_run(name="TODO", dataset=dataset, predictions=[])
        prediction_items = []
        for s3url, dataset_item in self.s3url_to_dataset_map.items():
            item_link = self.responses[s3url]
            # TODO download data at item_link
            item_link = [(100,100,500,500,0)]  # Temporary, hardcoded box
            # TODO convert the data into a Prediction object
            ref_id = dataset_item.reference_id
            for box in item_link:
                # TODO assuming box is a list of (x, y, w, h, label). This is probably not the case
                pred_item = nucleus.BoxPrediction(label=str(box[4]), x=box[0], y=box[1], width=box[2], height=box[3], reference_id=ref_id)
                prediction_items.append(pred_item)

        job = model_run.predict(prediction_items, asynchronous=True)
        job.sleep_until_complete()
        job.errors()


def _nucleus_ds_to_s3url_list(dataset: Dataset) -> Sequence[str]:
    # TODO I'm not sure if dataset items are necessarily s3URLs. Does this matter?
    # TODO support lidar point clouds
    if len(dataset.items) == 0:
        logger.warning("Passed a dataset of length 0")
        return None  # TODO return type?
    dataset_item_type = dataset.items[0].type
    if not all([data.type == dataset_item_type for data in dataset.items]):
        logger.warning("Dataset has multiple item types")
        raise Exception  # TODO (code style) too broad exception

    s3url_to_dataset_map = {}
    # Do we need to keep track of nucleus ids?
    if dataset_item_type == DatasetItemType.IMAGE:
        s3Urls = [data.image_location for data in dataset.items]
        s3url_to_dataset_map = {data.image_location: data for data in dataset.items}
    elif dataset_item_type == DatasetItemType.POINTCLOUD:
        s3Urls = [data.pointcloud_location for data in dataset.items]
        s3url_to_dataset_map = {data.pointcloud_location: data for data in dataset.items}
    else:
        raise NotImplementedError(f"Dataset Item Type {dataset_item_type} not implemented")
    # TODO for demo
    return s3Urls, s3url_to_dataset_map  # TODO duplicated data in returned values


class ModelEndpoint:
    """
    Represents an endpoint on Hosted Model Inference
    """
    def __init__(self, endpoint_name, endpoint_url):
        # TODO what are endpoint_name and endpoint_url?
        self.endpoint_name = endpoint_name
        self.endpoint_url = endpoint_url

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

        request_endpoint = f"task_async/{self.endpoint_name}"  # is endpoint_name correct?
        for s3url in s3urls:
            # payload = dict(img_url=s3url)  # TODO format idk
            payload = s3url
            # TODO make these requests in parallel instead of making them serially
            inference_request = make_hosted_inference_request(payload=payload, route=request_endpoint, requests_command=requests.post, use_json=False)  # Avoid using json because endpoint expects raw url
            request_ids[s3url] = inference_request['task_id']
            # make the request to the endpoint (in parallel or something)

        return ModelEndpointAsyncJob(request_ids=request_ids, model_endpoint=self, dataset_to_s3url_map=s3url_to_dataset_map)

    def status(self):
        # Makes call to model status endpoint
        raise NotImplementedError


def make_hosted_inference_request(
    payload: dict, route: str, requests_command=requests.post, use_json: bool=True
) -> dict:
    """
    Makes a request to Hosted Inference endpoint and logs a warning if not
    successful.

    :param payload: given payload
    :param route: route for the request
    :param requests_command: requests.post, requests.get, requests.delete
    :param use_json: whether we should use a json-formatted payload or not
    :return: response JSON
    """
    endpoint = f"{HOSTED_INFERENCE_ENDPOINT}/{route}"

    logger.info("Posting to %s", endpoint)

    if use_json:
        response = requests_command(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            # auth=(self.api_key, ""), # or something
            timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
        )
    else:
        response = requests_command(
            endpoint,
            data=payload,
            timeout=DEFAULT_NETWORK_TIMEOUT_SEC
        )
    logger.info("API request has response code %s", response.status_code)

    if not response.ok:
        logger.warning(f"Request failed at route {route}, payload {payload}, code {response.status_code}")
        raise Exception("Response was not ok")
    print(response)
    return response.json()


def make_multiple_hosted_inference_requests(payload_route_commands: Sequence[Tuple[dict, str, Callable]]):
    """
    Make multiple requests in parallel
    """
    # TODO make parallel requests
    raise NotImplementedError


# TODO: add these to __init__


def add_model_bundle(
    model_name: str, model: Any, load_predict_fn: Any, reference_id: str
):
    """
    Uploads to s3 (for now, will upload to s3 signed url later) a model bundle, i.e. a dictionary
    {
        "model": model
        "load_predict_fn": load_predict_fn
    }
    """
    # TODO: types of model and load_predict_fn
    # (Temporary) For now we do some s3 string manipulation, later on get an s3URL from some
    # getPresignedURL endpoint
    model_bundle_name = f"{model_name}_{reference_id}"
    s3_path = f"s3://scale-ml/hosted-model-inference/bundles/{model_bundle_name}.pkl"
    # this might be an invalid url but this is temporary anyways
    kwargs = {
        "transport_params": {"session": Session(profile_name="ml-worker")}
    }

    with smart_open.open(s3_path, "wb", **kwargs) as bundle_pkl:
        bundle = dict(model=model, load_predict_fn=load_predict_fn)
        # TODO does this produce a performance bottleneck
        # This might be a bit slow, the "correct" thing to do is probably to
        # dump the pickle locally, zip it, and upload the corresponding zip to s3
        # In any case, this is temporary.
        cloudpickle.dump(bundle, bundle_pkl)

        # TODO upload the file to a signed url

    # Make request to hosted inference service
    make_hosted_inference_request(
        dict(model_name=model_name, reference_id=reference_id),
        route="model-bundle",
    )

    return ModelBundle(model_bundle_name)


def create_model_endpoint(
    service_name: str,
    model_bundle: ModelBundle,
    cpus: int,
    memory: str,
    gpus: int,
    gpu_type: str,
    min_workers: int,
    max_workers: int,
    per_worker: int,
    requirements: Dict[str, str],
    env_params: Dict[str, str],
) -> ModelEndpoint:
    """
    requirements: A Dictionary containing package name -> version string for the endpoint.
    env_params: A Dictionary containing keys framework_type, pytorch_version, cuda_version, cudnn_version.
    """
    # TODO: input validation?
    # This should make an HTTP request to the Hosted Model Inference server at the "create model endpoint" endpoint
    payload = dict(
        service_name=service_name,
        env_params=env_params,
        bundle_id=model_bundle.name,
        cpus=cpus,
        memory=memory,
        gpus=gpus,
        gpu_type=gpu_type,
        min_workers=min_workers,
        max_workers=max_workers,
        per_worker=per_worker,
        requirements=requirements,
    )

    resp = make_hosted_inference_request(
        payload, "endpoints", requests_command=requests.post
    )

    # TODO what is the format of response?

    print("Temp resp format:", resp)

    endpoint_name = resp["endpoint_name"]
    endpoint_url = resp["endpoint_url"]

    return ModelEndpoint(endpoint_name, endpoint_url)
