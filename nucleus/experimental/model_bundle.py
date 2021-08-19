from nucleus.dataset import Dataset
from typing import Any
import tempfile
import dill
import smart_open


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
    """
    Uploads to s3 (for now, will upload to actual service later) a model bundle, i.e. a dictionary
    {
        "model": model
        "load_predict_fn": load_predict_fn
    }
    """
    # TODO: stub, types of model and load_predict_fn
    # For now we do some s3 string manipulation
    s3_path = f"s3://scale-ml/hosted-model-inference/bundles/{model_name}_{reference_id}.pkl"
    # this might be an invalid url but this is temporary anyways
    with smart_open.open(s3_path, "r") as bundle_pkl:
        bundle = dict(model=model, load_predict_fn=load_predict_fn)
        dill.dump(bundle, bundle_pkl)

        # TODO upload the file via http request later

    # TODO make request to hosted model inference (hmm how will that work?
    #  We probably want to abstract out the make_request thing but there's already some work inside this library)






    raise NotImplementedError


def create_model_endpoint(endpoint_name: str, model_bundle: ModelBundle, cpus: int, memory: str, gpus: int, gpu_type: str, sync_type: str, min_workers: int, max_workers: int):
    # TODO: stub
    # TODO: input validation?
    # This should make an HTTP request to the Hosted Model Inference server at the "create model endpoint" endpoint
    raise NotImplementedError
