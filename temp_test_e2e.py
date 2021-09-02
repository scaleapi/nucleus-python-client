from nucleus.experimental.model_bundle import *

# Don't include in pr

env_params = {
    "framework_type": "pytorch",
    "pytorch_version": "1.7.0",
    "cuda_version": "11.0",
    "cudnn_version": "cudnn8-devel",
}

mb = ModelBundle("abc123")

args = {
    "service_name": "seantest",
    "env_params": env_params,
    #"bundle_id": "abc123",
    "model_bundle": mb,
    "cpus": 1,
    "memory": "4Gi",
    "gpus": 1,
    "gpu_type": "nvidia-tesla-t4",
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    "requirements": [],
}

create_model_endpoint(**args)

