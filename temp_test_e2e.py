import os
import time

import nucleus
from nucleus.deploy.client import DeployClient
from nucleus.deploy.find_packages import find_packages_from_imports
from nucleus.deploy.model_bundle import ModelBundle

# TODO Don't include this file in final pr
from nucleus.deploy.model_endpoint import ModelEndpoint
from nucleus.deploy.nucleus_integration import (
    create_nucleus_dataset_inference_run,
)


def create_dummy_bundle(hmi_client):
    def returns_returns_1(x):
        def returns_1(y):
            return 1

        return returns_1

    model = None
    load_predict_func = returns_returns_1
    hmi_client.create_model_bundle("return1", model, load_predict_func)


def create_endpoint(hmi_client):
    env_params = {
        "framework_type": "pytorch",
        "pytorch_version": "1.7.0",
        "cuda_version": "11.0",
        "cudnn_version": "cudnn8-devel",
    }

    mb = ModelBundle("abc123")

    # TODO out of date lol
    args = {
        "service_name": "seantest",
        "env_params": env_params,
        # "bundle_id": "abc123",
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

    model_endpoint = hmi_client.create_model_endpoint(**args)
    print(model_endpoint.endpoint_id)


def make_task_call(
    endpoint_name: str, dataset_id: str, upload_to_nucleus=True
):
    print("Need to export NUCLEUS_API_KEY=live_<your live api key here>")
    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])

    dataset = client.get_dataset(dataset_id)

    model_endpoint = ModelEndpoint(
        endpoint_id=endpoint_name, client=hmi_client
    )

    inference_run = create_nucleus_dataset_inference_run(
        model_endpoint, client, dataset
    )

    while not inference_run.is_done(poll=True):
        print("Waiting for predictions to finish...")
        time.sleep(5)

    print("Predictions complete!")
    predictions = inference_run.hmi_async_job.get_responses()
    print(predictions)

    if upload_to_nucleus:
        ts = str(time.time())
        inference_run.upload_to_nucleus(
            model_run_name=ts,
            model_name="Test HMI upload",
            model_ref_id=f"test_hmi_upload_{ts}",
        )


def temp_clone_pandaset():
    print("Need to export NUCLEUS_API_KEY=live_<your live api key here>")
    public_dataset_client = nucleus.NucleusClient(
        os.environ["NUCLEUS_PUBLIC_DATASET_API_KEY"]
    )

    dataset = public_dataset_client.get_dataset(
        "ds_bwhjbyfb8mjj0ykagxf0"
    )  # Public Pandaset Dataset id
    dataset_items = dataset.items

    client = nucleus.NucleusClient(os.environ["NUCLEUS_API_KEY"])
    cloned_dataset = client.create_dataset("Pandaset clone - small")
    cloned_dataset_items = [
        nucleus.DatasetItem(
            image_location=x.image_location,
            reference_id=x.reference_id,
            metadata=x.metadata,
        )
        for x in dataset_items[:1]
    ]
    cloned_dataset.append(cloned_dataset_items)


# temp_clone_pandaset()

if __name__ == "__main__":
    # make_task_call(
    #     endpoint_name="yi-tf-test",
    #     #dataset_id="ds_c4wht080x81g060m0nfg",
    #     dataset_id="ds_c4x8s3m6n260060cngs0",
    #     upload_to_nucleus=True
    # )
    # temp_clone_pandaset()

    packages = find_packages_from_imports(globals())
    print(packages)
    packages = find_packages_from_imports(globals())
    print(packages)

    hmi_client = DeployClient(api_key=os.environ["NUCLEUS_API_KEY"])
    img_url = (
        "https://scale.com/_next/static/media/dashboard-hero.ab478d39.png"
    )
    # hmi_client.create_endpoint()
    # print(hmi_client.connection.post({}, "model_bundle_upload"))
    # create_dummy_bundle(hmi_client)
    env_params = {
        "framework_type": "pytorch",
        "pytorch_version": "1.7.0",
        "cuda_version": "11.0",
        "cudnn_version": "cudnn8-devel",
    }
    # me2 = hmi_client.create_model_endpoint(
    #     service_name="seantest2",
    #     env_params=env_params,
    #     model_bundle=ModelBundle(name="return1"),
    #     cpus=1,
    #     memory="4Gi",
    #     gpus=0,
    #     gpu_type="Doesn't Matter",
    #     min_workers=1,
    #     max_workers=1,
    #     per_worker=1,
    #     requirements=[],
    # )
    # print(me2)

    mes = hmi_client.list_model_endpoints()
    for me in mes:
        print(me)
    m1 = mes[0]
    print(hmi_client.sync_request("seantest2", img_url))
    async_task = hmi_client.async_request("seantest2", img_url)
    print(async_task)
    print(hmi_client.get_async_response(async_task))
    for i in range(10):
        time.sleep(1)
        print(hmi_client.get_async_response(async_task))
    # print(hmi_client.get_bundles())