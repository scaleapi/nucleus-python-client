import time
import os

from nucleus.experimental.DEPRECATED_model_bundle import *
# TODO Don't include this file in final pr
from nucleus.experimental.model_endpoint import ModelEndpoint, ModelBundle
from nucleus.experimental.hosted_inference_client import HostedInference

def create_endpoint():
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

    model_endpoint = create_model_endpoint(**args)
    print(model_endpoint.endpoint_url, model_endpoint.endpoint_id)


def make_task_call(endpoint_name: str, dataset_id: str, upload_to_nucleus = True):
    print("Need to export NUCLEUS_API_KEY=live_<your live api key here>")
    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])

    dataset = client.get_dataset(dataset_id)

    model_endpoint = ModelEndpoint(endpoint_name=endpoint_name, endpoint_url="idk")
    model_endpoint_async_job = model_endpoint.create_run_job(dataset=dataset)

    while not model_endpoint_async_job.is_done(poll=True):
        print("Waiting for predictions to finish...")
        time.sleep(5)

    print("Predictions complete!")
    predictions = model_endpoint_async_job.get_responses()
    print(predictions)

    if upload_to_nucleus:
        ts = str(time.time())
        model_endpoint_async_job.upload_responses_to_nucleus(client, dataset, ts, model_name="Test HMI upload", model_ref_id=f"test_hmi_upload_{ts}")


def temp_clone_pandaset():
    print("Need to export NUCLEUS_API_KEY=live_<your live api key here>")
    public_dataset_client = nucleus.NucleusClient(os.environ["NUCLEUS_PUBLIC_DATASET_API_KEY"])

    dataset = public_dataset_client.get_dataset("ds_bwhjbyfb8mjj0ykagxf0")  # Public Pandaset Dataset id
    dataset_items = dataset.items

    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])
    cloned_dataset = client.create_dataset("Pandaset clone - small")
    cloned_dataset_items = [
        nucleus.DatasetItem(
            image_location=x.image_location, reference_id=x.reference_id, metadata=x.metadata
        ) for x in dataset_items[:1]
    ]
    cloned_dataset.append(cloned_dataset_items)


# create_endpoint()

#temp_clone_pandaset()

if __name__ == "__main__":
    # make_task_call(
    #     endpoint_name="yi-tf-test",
    #     #dataset_id="ds_c4wht080x81g060m0nfg",
    #     dataset_id="ds_c4x8s3m6n260060cngs0",
    #     upload_to_nucleus=True
    # )
    #temp_clone_pandaset()

    hmi_client = HostedInference(api_key="")
    # print(hmi_client.connection.post({}, "model_bundle_upload"))
    print(hmi_client.connection.get("endpoints"))
