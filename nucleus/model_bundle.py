from .dataset import Dataset
from typing import Any
import tempfile
import dill


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


# TODO: add these to __init__

def add_model_bundle(model_name: str, model: Any, load_predict_fn: Any, reference_id: str):
    # TODO: stub, types of model and load_predict_fn


    raise NotImplementedError


def create_model_endpoint(endpoint_name: str, model_bundle: ModelBundle, cpus: int, memory: str, gpus: int, gpu_type: str, sync_type: str, min_workers: int, max_workers: int):
    # TODO: stub
    # TODO: input validation?
    # This should make an HTTP request to the Hosted Model Inference server at the "create model endpoint" endpoint
    raise NotImplementedError
