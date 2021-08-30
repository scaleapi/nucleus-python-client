import logging
from typing import Any, Dict

import cloudpickle
import requests
import smart_open
from boto3 import Session

from nucleus.dataset import Dataset

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

        # TODO upload the file via http request later

    # Make request to hosted inference service
    make_hosted_inference_request(
        dict(model_name=model_name, reference_id=reference_id),
        route="model-bundle",
    )

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
    per_worker: int,
    requirements: Dict[str, str],
):
    # TODO: stub
    # TODO: input validation?
    # This should make an HTTP request to the Hosted Model Inference server at the "create model endpoint" endpoint
    env_params = requirements  # TODO how do we get env_params?
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

    return make_hosted_inference_request(
        payload, "endpoints", requests_command=requests.post
    )
