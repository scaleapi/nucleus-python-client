import logging
from typing import Any, Dict

import cloudpickle
import requests
import smart_open
from boto3 import Session

from nucleus.dataset import Dataset
from nucleus.dataset_item import DatasetItemType

# TODO temporary endpoint, will be replaced with some https://api.scale.com/hostedinference/<sub-route>
HOSTED_INFERENCE_ENDPOINT = "http://hostedinference.ml-staging-internal.scale.com"  # TODO this isn't https
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
    # TODO Everything
    def __init__(self):
        pass

    def is_done(self):
        # TODO: make some request to some endpoint
        raise NotImplementedError


class ModelEndpoint:
    """
    Represents an endpoint on Hosted Model Inference
    """
    def __init__(self, endpoint_name, endpoint_url):
        self.endpoint_name = endpoint_name
        self.endpoint_url = endpoint_url

    def create_run_job(self, model_name: str, dataset: Dataset):
        # TODO: stub
        # TODO: take the dataset, translate to s3URLs

        # TODO support lidar point clouds
        if len(dataset.items) == 0:
            logger.warning("Passed a dataset of length 0")
            return None  # TODO return type?
        dataset_item_type = dataset.items[0].type
        if not all([data.type == dataset_item_type for data in dataset.items]):
            logger.warning("Dataset has multiple item types")
            raise Exception  # TODO too broad exception

        # Do we need to keep track of nucleus ids?
        if dataset_item_type == DatasetItemType.IMAGE:
            s3URLs = [data.image_location for data in dataset.items]
        elif dataset_item_type == DatasetItemType.POINTCLOUD:
            s3URLs = [data.pointcloud_location for data in dataset.items]

        # TODO: pass s3URLs to some run job creation endpoint
        # payload = {"model_name": model_name}
        # make_hosted_inference_request()

        # return ModelEndpointAsyncJob

        raise NotImplementedError

    def status(self):
        # Makes call to model status endpoint
        raise NotImplementedError


def make_hosted_inference_request(
    payload: dict, route: str, requests_command=requests.post
) -> dict:
    """
    Makes a request to Hosted Inference endpoint and logs a warning if not
    successful.

    :param payload: given payload
    :param route: route for the request
    :param requests_command: requests.post, requests.get, requests.delete
    :return: response JSON
    """
    endpoint = f"{HOSTED_INFERENCE_ENDPOINT}/{route}"

    logger.info("Posting to %s", endpoint)

    response = requests_command(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        # auth=(self.api_key, ""), # or something
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    logger.info("API request has response code %s", response.status_code)

    if not response.ok:
        raise Exception("Response was not ok")

    return response.json()


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
    # For now we do some s3 string manipulation
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
    endpoint_name: str,
    model_bundle: ModelBundle,
    cpus: int,
    memory: str,
    gpus: int,
    gpu_type: str,
    sync_type: str,
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
    # TODO: stub
    # TODO: input validation?
    # This should make an HTTP request to the Hosted Model Inference server at the "create model endpoint" endpoint
    payload = dict(
        service_name=endpoint_name,
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

    endpoint_name = resp["endpoint_name"]
    endpoint_url = resp["endpoint_url"]

    return ModelEndpoint(endpoint_name, endpoint_url)  # TODO what is the format of response?
