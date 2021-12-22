from typing import Any

import cloudpickle
import logging
import requests
from typing import Dict

from nucleus.connection import Connection
from nucleus.experimental.model_endpoint import ModelEndpoint, ModelBundle

HOSTED_INFERENCE_ENDPOINT = "https://api.scale.com/hosted_inference"
DEFAULT_NETWORK_TIMEOUT_SEC = 120

logger = logging.getLogger(__name__)
logging.basicConfig()

class HostedInference:
    """HostedInference Python Client extension."""

    def __init__(self, api_key: str, endpoint: str):
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
        if "signed_url" not in model_bundle_s3_url:
            raise Exception("Error in server request, no signedURL found")  # TODO code style broad exception
        s3_path = model_bundle_s3_url["signed_url"]
        raw_s3_url = f"s3://{model_bundle_s3_url['bucket']}/{model_bundle_s3_url['key']}"

        # Make bundle upload
        bundle = dict(model=model, load_predict_fn=load_predict_fn)
        serialized_bundle = cloudpickle.dumps(bundle)
        requests.put(s3_path, data=serialized_bundle)

        self.connection.post(payload=dict(id=model_bundle_name, location=raw_s3_url), route="model_bundle")
        # TODO check that a model bundle was created and no name collisions happened
        return ModelBundle(model_bundle_name)

    def create_model_endpoint(self,
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
                              ):
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
        resp = self.connection.post(payload, "endpoints")
        endpoint_id = resp["endpoint_id"]
        return ModelEndpoint(endpoint_id=endpoint_id)
