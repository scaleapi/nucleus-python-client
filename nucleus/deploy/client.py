import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar

import cloudpickle
import requests

from nucleus.connection import Connection
from nucleus.deploy.constants import (
    ASYNC_TASK_PATH,
    ASYNC_TASK_RESULT_PATH,
    ENDPOINT_PATH,
    MODEL_BUNDLE_SIGNED_URL_PATH,
    SCALE_DEPLOY_ENDPOINT,
    SYNC_TASK_PATH,
)
from nucleus.deploy.find_packages import find_packages_from_imports
from nucleus.deploy.model_bundle import ModelBundle
from nucleus.deploy.model_endpoint import AsyncModelEndpoint

DEFAULT_NETWORK_TIMEOUT_SEC = 120

logger = logging.getLogger(__name__)
logging.basicConfig()

DeployModel_T = TypeVar("DeployModel_T")


class DeployClient:
    """Scale Deploy Python Client extension."""

    def __init__(self, api_key: str, endpoint: str = SCALE_DEPLOY_ENDPOINT):
        """
        Initializes a Scale Deploy Client.

        Parameters:
            api_key: Your Scale API key
            endpoint: The Scale Deploy Endpoint (this should not need to be changed)
        """
        self.connection = Connection(api_key, endpoint)

    def __repr__(self):
        return f"DeployClient(connection='{self.connection}')"

    def __eq__(self, other):
        return self.connection == other.connection

    def create_model_bundle(
        self,
        model_bundle_name: str,
        load_predict_fn: Callable[[DeployModel_T], Callable[[Any], Any]],
        model: Optional[DeployModel_T] = None,
        load_model_fn: Optional[Callable[[], DeployModel_T]] = None,
    ) -> ModelBundle:
        """
        Grabs a s3 signed url and uploads a model bundle to Scale Deploy.
        A model bundle consists of a "load_predict_fn" and exactly one of "model" or "load_model_fn", such that
        load_predict_fn(model)
        or
        load_predict_fn(load_model_fn())
        returns a function predict_fn that takes in model input and returns model output.
        Pre/post-processing code can be included inside load_predict_fn/model.

        Parameters:
            model_bundle_name: Name of model bundle you want to create. This acts as a unique identifier.
            model: Typically a trained Neural Network, e.g. a Pytorch module
            load_model_fn: Function that when run, loads a model, e.g. a Pytorch module
            load_predict_fn: Function that when called with model, returns a function that carries out inference
        """

        if (model is not None and load_model_fn is not None) or (
            model is None and load_model_fn is None
        ):
            raise ValueError(
                "Exactly one of model and load_model_fn should be non-None"
            )
        # TODO should we try to catch when people intentionally pass both model and load_model_fn as None?

        # Grab a signed url to make upload to
        model_bundle_s3_url = self.connection.post(
            {}, MODEL_BUNDLE_SIGNED_URL_PATH
        )
        s3_path = model_bundle_s3_url["signedUrl"]
        raw_s3_url = f"s3://{model_bundle_s3_url['bucket']}/{model_bundle_s3_url['key']}"

        # Make bundle upload
        if model is not None:
            bundle = dict(model=model, load_predict_fn=load_predict_fn)
        else:
            bundle = dict(
                load_model_fn=load_model_fn, load_predict_fn=load_predict_fn
            )
        serialized_bundle = cloudpickle.dumps(bundle)
        requests.put(s3_path, data=serialized_bundle)

        self.connection.post(
            payload=dict(id=model_bundle_name, location=raw_s3_url),
            route="model_bundle",
        )  # TODO use return value somehow
        # resp["data"]["bundle_name"] should equal model_bundle_name
        # TODO check that a model bundle was created and no name collisions happened
        return ModelBundle(model_bundle_name)

    def create_model_endpoint(
        self,
        service_name: str,
        model_bundle: ModelBundle,
        cpus: int,
        memory: str,
        gpus: int,
        min_workers: int,
        max_workers: int,
        per_worker: int,
        env_params: Dict[str, str],
        requirements: Optional[List[str]] = None,
        gpu_type: Optional[str] = None,
    ) -> AsyncModelEndpoint:
        """
        Creates a Model Endpoint that is able to serve requests

        Parameters:
            service_name: Name of model endpoint. Must be unique.
            model_bundle: The ModelBundle that you want your Model Endpoint to serve
            cpus: Number of cpus each worker should get, e.g. 1, 2, etc.
            memory: Amount of memory each worker should get, e.g. "4Gi", "512Mi", etc.
            gpus: Number of gpus each worker should get, e.g. 0, 1, etc.
            min_workers: Minimum number of workers for model endpoint
            max_workers: Maximum number of workers for model endpoint
            per_worker: An autoscaling parameter. Use this to make a tradeoff between latency and costs,
                a lower per_worker will mean more workers are created for a given workload
            requirements: A list of python package requirements, e.g.
                ["tensorflow==2.3.0", "tensorflow-hub==0.11.0"]. If no list has been passed, will default to the currently
                imported list of packages.
            env_params: A dictionary that dictates environment information e.g.
                the use of pytorch or tensorflow, which cuda/cudnn versions to use.
                Specifically, the dictionary should contain the following keys:
                "framework_type": either "tensorflow" or "pytorch".
                "pytorch_version": Version of pytorch, e.g. "1.5.1", "1.7.0", etc. Only applicable if framework_type is pytorch
                "cuda_version": Version of cuda used, e.g. "11.0".
                "cudnn_version" Version of cudnn used, e.g. "cudnn8-devel".
                "tensorflow_version": Version of tensorflow, e.g. "2.3.0". Only applicable if framework_type is tensorflow
            gpu_type: If specifying a non-zero number of gpus, this controls the type of gpu requested. Current options are
                "nvidia-tesla-t4" for NVIDIA T4s, or "nvidia-tesla-v100" for NVIDIA V100s.

        Returns:
             A ModelEndpoint object that can be used to make requests to the endpoint.

        """
        if requirements is None:
            requirements_inferred = find_packages_from_imports(globals())
            requirements = [
                f"{key}=={value}"
                for key, value in requirements_inferred.items()
            ]
            logger.info(
                "Using \n%s\n for model endpoint %s",
                requirements,
                service_name,
            )
            # TODO test
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
            del payload["gpu_type"]
        elif gpus > 0 and gpu_type is None:
            raise ValueError("If nonzero gpus, must provide gpu_type")
        resp = self.connection.post(payload, ENDPOINT_PATH)
        endpoint_creation_task_id = resp[
            "endpoint_id"
        ]  # Serverside needs updating
        logger.info(
            "Endpoint creation task id is %s", endpoint_creation_task_id
        )
        return AsyncModelEndpoint(endpoint_id=service_name, client=self)

    # Relatively small wrappers around http requests

    def list_bundles(self) -> List[ModelBundle]:
        """
        Returns a list of model bundles that the user owns.
        TODO this route doesn't exist serverside
        """
        # resp = self.connection.get("model_bundle")
        raise NotImplementedError

    def list_model_endpoints(self) -> List[AsyncModelEndpoint]:
        """
        Lists all model endpoints that the user owns.
        TODO: single get_model_endpoint(self)? route doesn't exist serverside I think

        Returns:
            A list of ModelEndpoint objects
        """
        resp = self.connection.get(ENDPOINT_PATH)
        return [
            AsyncModelEndpoint(endpoint_id=endpoint_id, client=self)
            for endpoint_id in resp
        ]

    def sync_request(self, endpoint_id: str, url: str) -> str:
        """
        DEPRECATED
        Makes a request to the Model Endpoint at endpoint_id, and blocks until request completion or timeout.

        Parameters:
            endpoint_id: The id of the endpoint to make the request to
            url: A url that points to a file containing model input.
                Must be accessible by Scale Deploy, hence it needs to either be public or a signedURL.

        Returns:
            A signedUrl that contains a cloudpickled Python object, the result of running inference on the model input
            Example output:
                `https://foo.s3.us-west-2.amazonaws.com/bar/baz/qux?xyzzy`
        """
        resp = self.connection.post(
            payload=dict(url=url), route=f"{SYNC_TASK_PATH}/{endpoint_id}"
        )
        return resp["result_url"]

    def async_request(self, endpoint_id: str, url: str) -> str:
        """
        Not recommended to use this, instead we recommend to use functions provided by AsyncModelEndpoint.
        Makes a request to the Model Endpoint at endpoint_id, and immediately returns a key that can be used to retrieve
        the result of inference at a later time.

        Parameters:
            endpoint_id: The id of the endpoint to make the request to
            url: A url that points to a file containing model input.
                Must be accessible by Scale Deploy, hence it needs to either be public or a signedURL.

        Returns:
            An id/key that can be used to fetch inference results at a later time.
            Example output:
                `abcabcab-cabc-abca-0123456789ab`
        """
        resp = self.connection.post(
            payload=dict(url=url), route=f"{ASYNC_TASK_PATH}/{endpoint_id}"
        )
        return resp["task_id"]

    def get_async_response(self, async_task_id: str) -> str:
        """
        Not recommended to use this, instead we recommend to use functions provided by AsyncModelEndpoint.
        Gets inference results from a previously created task.

        Parameters:
            async_task_id: The id/key returned from a previous invocation of async_request.

        Returns:
            A dictionary that contains task status and optionally a result url if the task has completed.
            Dictionary's keys are as follows:
            state: 'PENDING' or 'SUCCESS' or 'FAILURE'
            result_url: a url pointing to inference results. This url is accessible for 12 hours after the request has been made.
            Example output:
                `{'state': 'SUCCESS', 'result_url': 'https://foo.s3.us-west-2.amazonaws.com/bar/baz/qux?xyzzy'}`
        TODO: do we want to read the results from here as well? i.e. translate result_url into a python object
        """

        resp = self.connection.get(
            route=f"{ASYNC_TASK_RESULT_PATH}/{async_task_id}"
        )
        return resp

    def batch_async_request(self, endpoint_id: str, urls: List[str]):
        """
        Sends a batch inference request to the Model Endpoint at endpoint_id, returns a key that can be used to retrieve
        the results of inference at a later time.

        Parameters:
            endpoint_id: The id of the endpoint to make the request to
            urls: A list of urls, each pointing to a file containing model input.
                Must be accessible by Scale Deploy, hence urls need to either be public or signedURLs.

        Returns:
            An id/key that can be used to fetch inference results at a later time
        """
        raise NotImplementedError

    def get_batch_async_response(self, batch_async_task_id: str):
        """
        TODO not sure about how the batch task returns an identifier for the batch.
        Gets inference results from a previously created batch task.

        Parameters:
            batch_async_task_id: An id representing the batch task job

        Returns:
            TODO Something similar to a list of signed s3URLs
        """
        raise NotImplementedError
