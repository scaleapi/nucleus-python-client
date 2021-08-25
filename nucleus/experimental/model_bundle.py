from nucleus.dataset import Dataset
from typing import Any
import tempfile
import logging
import dill
import smart_open
import requests
from boto3 import Session

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


class ModelEndpoint:
    def __init__(self):
        # TODO: stub
        pass

    def create_run_job(self, model_name: str, dataset: Dataset):
        # TODO: stub
        raise NotImplementedError


def make_hosted_inference_request(
    payload: dict, route: str, requests_command=requests.post
) -> dict:
    """
    Makes a request to Nucleus endpoint and logs a warning if not
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
    Uploads to s3 (for now, will upload to actual service later) a model bundle, i.e. a dictionary
    {
        "model": model
        "load_predict_fn": load_predict_fn
    }
    """
    # TODO: types of model and load_predict_fn
    # For now we do some s3 string manipulation
    s3_path = f"s3://scale-ml/hosted-model-inference/bundles/{model_name}_{reference_id}.pkl"
    # this might be an invalid url but this is temporary anyways
    kwargs = dict()

    kwargs["transport_params"] = {"session": Session(profile_name="ml-worker")}
    with smart_open.open(s3_path, "wb", **kwargs) as bundle_pkl:
        bundle = dict(model=model, load_predict_fn=load_predict_fn)
        # TODO does this produce a performance bottleneck
        dill.dump(bundle, bundle_pkl, recurse=True)

        # TODO upload the file via http request later

    # TODO make request to hosted model inference (hmm how will that work?
    #  We probably want to abstract out the make_request thing but there's already some work inside this library)
    model_bundle_name = make_hosted_inference_request(
        dict(model_name=model_name, reference_id=reference_id),
        route="model-bundle/create",
    )["model_bundle"]

    return ModelBundle(model_bundle_name)

    # raise NotImplementedError


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
):
    # TODO: stub
    # TODO: input validation?
    # This should make an HTTP request to the Hosted Model Inference server at the "create model endpoint" endpoint
    raise NotImplementedError
