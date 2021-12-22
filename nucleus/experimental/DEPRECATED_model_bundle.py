import logging
from typing import Callable, Dict, Sequence, Tuple, Any

import cloudpickle
import requests

from nucleus.experimental.hosted_inference_client import HOSTED_INFERENCE_ENDPOINT
from nucleus.experimental.model_endpoint import ModelEndpoint, ModelBundle

DEFAULT_NETWORK_TIMEOUT_SEC = 120

logger = logging.getLogger(__name__)
logging.basicConfig()


def make_multiple_hosted_inference_requests(payload_route_commands: Sequence[Tuple[dict, str, Callable]]):
    """
    Make multiple requests in parallel
    """
    # TODO make parallel requests
    raise NotImplementedError


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
    TODO deprecated
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


def make_hosted_inference_request(
    payload: dict, route: str, requests_command=requests.post, use_json: bool=True
) -> dict:
    """
    TODO deprecated
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
            # auth=(self.api_key, ""), # TODO add this
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


def add_model_bundle(
    model_name: str, model: Any, load_predict_fn: Any, reference_id: str
):

    # TODO delete this, functionality is replaced
    # TODO: types of model and load_predict_fn

    # Grab a signed url to make upload to
    model_bundle_s3_url = make_hosted_inference_request({}, "model_bundle_upload", requests_command=requests.post)
    if "signed_url" not in model_bundle_s3_url:
        raise Exception("Error in server request, no signedURL found")  # TODO code style broad exception
    s3_path = model_bundle_s3_url["signed_url"]
    raw_s3_url = f"s3://{model_bundle_s3_url['bucket']}/{model_bundle_s3_url['key']}"

    # Make bundle upload
    bundle = dict(model=model, load_predict_fn=load_predict_fn)
    serialized_bundle = cloudpickle.dumps(bundle)
    requests.put(s3_path, data=serialized_bundle)

    # Make request to hosted inference service to save entry in database
    make_hosted_inference_request(
        dict(id=model_name, location=raw_s3_url),  # TODO model_name might not be the right id?
        route="model_bundle",
        requests_command=requests.post,
    )

    return ModelBundle(f"{model_name}_{reference_id}")  # TODO ModelBundleName is very wrong