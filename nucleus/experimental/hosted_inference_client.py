from typing import Any, Dict, List, Optional

import cloudpickle
import logging
import requests

from nucleus.connection import Connection
from nucleus.experimental.model_endpoint import ModelEndpoint, ModelBundle

HOSTED_INFERENCE_ENDPOINT = "https://api.scale.com/v1/hosted_inference"
DEFAULT_NETWORK_TIMEOUT_SEC = 120

logger = logging.getLogger(__name__)
logging.basicConfig()


class HostedInference:
    """HostedInference Python Client extension."""

    def __init__(self, api_key: str, endpoint: str = HOSTED_INFERENCE_ENDPOINT):
        self.connection = Connection(api_key, endpoint)

    def __repr__(self):
        return f"HostedInference(connection='{self.connection}')"

    def __eq__(self, other):
        return self.connection == other.connection

    def add_model_bundle(self, model_bundle_name: str, model: Any, load_predict_fn: Any):
        """
        Grabs a s3 signed url and uploads a model bundle, i.e. a dictionary
        {
            "model": model
            "load_predict_fn": load_predict_fn
        }
        """
        # Grab a signed url to make upload to
        model_bundle_s3_url = self.connection.post({}, "model_bundle_upload")
        if "signedUrl" not in model_bundle_s3_url:
            raise Exception("Error in server request, no signedURL found")  # TODO code style broad exception
        s3_path = model_bundle_s3_url["signedUrl"]
        raw_s3_url = f"s3://{model_bundle_s3_url['bucket']}/{model_bundle_s3_url['key']}"

        # Make bundle upload
        bundle = dict(model=model, load_predict_fn=load_predict_fn)
        serialized_bundle = cloudpickle.dumps(bundle)
        requests.put(s3_path, data=serialized_bundle)

        resp = self.connection.post(payload=dict(id=model_bundle_name, location=raw_s3_url), route="model_bundle")
        # resp["data"]["bundle_name"] should equal model_bundle_name
        # TODO check that a model bundle was created and no name collisions happened
        return ModelBundle(model_bundle_name)

    # TODO some function to get requirements

    def create_model_endpoint(self,
                              service_name: str,
                              model_bundle: ModelBundle,
                              cpus: int,
                              memory: str,
                              gpus: int,
                              min_workers: int,
                              max_workers: int,
                              per_worker: int,
                              requirements: List[str],  # Dict[str, str],
                              env_params: Dict[str, str],
                              gpu_type: Optional[str] = None,
                              ):
        payload = dict(
            service_name=service_name,
            env_params=env_params,
            bundle_name=model_bundle.name,
            cpus=cpus,
            memory=memory,
            gpus=gpus,
            gpu_type=gpu_type,
            min_workers=min_workers,
            max_workers=max_workers,
            per_worker=per_worker,
            requirements=requirements,
        )
        if gpus == 0:
            del payload['gpu_type']
        elif gpus > 0 and gpu_type is None:
            raise ValueError("If nonzero gpus, must provide gpu_type")
        resp = self.connection.post(payload, "endpoints")
        endpoint_id = resp["data"]["endpoint_id"]  # TODO this is very wrong
        return ModelEndpoint(endpoint_id=endpoint_id, client=self)

    # Relatively small wrappers around http requests

    def get_bundles(self) -> List[ModelBundle]:
        # TODO this route currently doesn't exist serverside
        resp = self.connection.get("model_bundle")
        raise NotImplementedError

    def get_model_endpoints(self) -> List[ModelEndpoint]:
        resp = self.connection.get("endpoints")
        return [ModelEndpoint(endpoint_id=endpoint_id, client=self) for endpoint_id in resp]

    def edit_model_endpoint(self):
        # TODO args, corresponds to PUT model_endpoint, doesn't exist serverside
        raise NotImplementedError

    def sync_request(self, endpoint_id: str, s3url: str):
        resp = self.connection.post(payload=dict(url=s3url), route=f"task/{endpoint_id}")
        return resp["data"]["result_url"]

    def async_request(self, endpoint_id: str, s3url: str):
        resp = self.connection.post(payload=dict(url=s3url), route=f"task_async/{endpoint_id}")
        return resp["data"]["task_id"]

    def get_async_response(self, async_task_id: str):

        resp = self.connection.get(route=f"task/result/{async_task_id}")
        return resp["data"]
